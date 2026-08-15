import { performance } from "node:perf_hooks";

import { isJSONRPCRequest, type JSONRPCMessage, type JSONRPCRequest } from "@modelcontextprotocol/sdk/types.js";

import { MockTransport, type MockState } from "./mockTransport.js";
import type { ToolCall, Trace } from "./types.js";

type MockResponses = ConstructorParameters<typeof MockTransport>[0];

interface AgentRunResult {
  finalOutput?: string;
  output?: string;
  text?: string;
  response?: string;
  totalTokensConsumed?: number;
  usage?: {
    totalTokens?: number;
    total_tokens?: number;
    inputTokens?: number;
    outputTokens?: number;
  };
}

interface AgentInstance {
  run?: (input: string, options?: Record<string, unknown>) => Promise<unknown> | unknown;
  invoke?: (input: string, options?: Record<string, unknown>) => Promise<unknown> | unknown;
  execute?: (input: string, options?: Record<string, unknown>) => Promise<unknown> | unknown;
  start?: (options?: Record<string, unknown>) => Promise<unknown> | unknown;
}

type AgentFactory =
  | AgentInstance
  | ((options: AgentBootstrapOptions) => AgentInstance | Promise<AgentInstance>)
  | (new (options: AgentBootstrapOptions) => AgentInstance);

interface AgentBootstrapOptions {
  transport: MockTransport;
  mockResponses: MockResponses;
  initialState?: Partial<MockState> | undefined;
}

interface AgentLoopErrorData {
  maxSteps: number;
  observedSteps: number;
}

export class AgentLoopException extends Error {
  public readonly maxSteps: number;
  public readonly observedSteps: number;

  public constructor(data: AgentLoopErrorData) {
    super(`Agent exceeded maxSteps (${data.maxSteps}) with ${data.observedSteps} tool calls.`);
    this.name = "AgentLoopException";
    this.maxSteps = data.maxSteps;
    this.observedSteps = data.observedSteps;
  }
}

export async function runPrompt(
  agent: AgentFactory,
  input: string,
  maxSteps = 15,
): Promise<Trace> {
  if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
    throw new Error("maxSteps must be a positive integer.");
  }

  const mockResponses = readOptionalProperty<MockResponses>(agent, "mockResponses") ?? {};
  const initialState = readOptionalProperty<Partial<MockState>>(agent, "initialState");

  const transport = new MockTransport(mockResponses, initialState);
  const trace: Trace = {
    initialUserPrompt: input,
    trajectory: [],
    totalTokensConsumed: 0,
    finalOutput: "",
  };

  let observedSteps = 0;
  const originalSend = transport.send.bind(transport);

  transport.send = async (message: JSONRPCMessage, options) => {
    const currentToolCall = parseToolCall(message);
    const startedAt = performance.now();

    if (currentToolCall) {
      observedSteps += 1;
      if (observedSteps > maxSteps) {
        throw new AgentLoopException({
          maxSteps,
          observedSteps,
        });
      }
    }

    try {
      await originalSend(message, options);
    } finally {
      if (currentToolCall) {
        const elapsedMs = Math.max(0, Math.round(performance.now() - startedAt));
        const trajectoryCall: ToolCall = {
          ...currentToolCall,
          latencyMs: elapsedMs,
        };
        trace.trajectory.push(trajectoryCall);
      }
    }
  };

  await transport.start();

  try {
    const instance = await instantiateAgent(agent, {
      transport,
      mockResponses,
      initialState,
    });

    const result = await executeAgent(instance, input, transport);
    trace.totalTokensConsumed = extractTotalTokens(result);
    trace.finalOutput = extractFinalOutput(result);

    return trace;
  } catch (error) {
    await transport.close();
    throw error;
  } finally {
    if (trace.finalOutput.length > 0) {
      await transport.close();
    }
  }
}

function parseToolCall(message: JSONRPCMessage): Omit<ToolCall, "latencyMs"> | undefined {
  if (!isJSONRPCRequest(message) || message.method !== "tools/call") {
    return undefined;
  }

  const toolName = extractToolName(message);
  if (!toolName) {
    return undefined;
  }

  return {
    toolName,
    arguments: extractToolArguments(message),
  };
}

async function instantiateAgent(agent: AgentFactory, options: AgentBootstrapOptions): Promise<AgentInstance> {
  if (isAgentInstance(agent)) {
    return agent;
  }

  if (typeof agent !== "function") {
    throw new Error("Unsupported agent target passed to runPrompt.");
  }

  try {
    const constructed = Reflect.construct(agent as new (opts: AgentBootstrapOptions) => AgentInstance, [options]);
    if (isAgentInstance(constructed)) {
      return constructed;
    }
  } catch {
    // Fall through and attempt functional factory invocation.
  }

  const factoryResult = await (agent as (opts: AgentBootstrapOptions) => AgentInstance | Promise<AgentInstance>)(options);
  if (!isAgentInstance(factoryResult)) {
    throw new Error("Agent factory did not return a valid agent instance.");
  }

  return factoryResult;
}

async function executeAgent(instance: AgentInstance, input: string, transport: MockTransport): Promise<unknown> {
  const runOptions = { transport };

  if (instance.run) {
    return instance.run(input, runOptions);
  }

  if (instance.invoke) {
    return instance.invoke(input, runOptions);
  }

  if (instance.execute) {
    return instance.execute(input, runOptions);
  }

  if (instance.start) {
    return instance.start({ input, ...runOptions });
  }

  throw new Error("Agent instance does not expose run, invoke, execute, or start.");
}

function extractToolName(request: JSONRPCRequest): string | undefined {
  const params = toRecord(request.params);
  if (!params) {
    return undefined;
  }

  return typeof params.name === "string" ? params.name : undefined;
}

function extractToolArguments(request: JSONRPCRequest): Record<string, unknown> {
  const params = toRecord(request.params);
  if (!params) {
    return {};
  }

  return toRecord(params.arguments) ?? {};
}

function extractTotalTokens(result: unknown): number {
  if (typeof result === "number") {
    return result;
  }

  const payload = toRecord(result);
  if (!payload) {
    return 0;
  }

  if (typeof payload.totalTokensConsumed === "number") {
    return payload.totalTokensConsumed;
  }

  const usage = toRecord(payload.usage);
  if (!usage) {
    return 0;
  }

  if (typeof usage.totalTokens === "number") {
    return usage.totalTokens;
  }

  if (typeof usage.total_tokens === "number") {
    return usage.total_tokens;
  }

  const inputTokens = typeof usage.inputTokens === "number" ? usage.inputTokens : 0;
  const outputTokens = typeof usage.outputTokens === "number" ? usage.outputTokens : 0;

  return inputTokens + outputTokens;
}

function extractFinalOutput(result: unknown): string {
  if (typeof result === "string") {
    return result;
  }

  const payloadRecord = toRecord(result);
  const payload = payloadRecord as AgentRunResult | undefined;
  if (!payload) {
    return "";
  }

  const textCandidate = payload.finalOutput ?? payload.output ?? payload.text ?? payload.response;
  if (typeof textCandidate === "string") {
    return textCandidate;
  }

  const message = toRecord(payloadRecord?.["message"]);
  if (message && typeof message.content === "string") {
    return message.content;
  }

  return "";
}

function isAgentInstance(value: unknown): value is AgentInstance {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as AgentInstance;
  return Boolean(candidate.run || candidate.invoke || candidate.execute || candidate.start);
}

function toRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

function readOptionalProperty<T>(value: unknown, key: string): T | undefined {
  if (!value || (typeof value !== "object" && typeof value !== "function")) {
    return undefined;
  }

  const candidate = value as Record<string, unknown>;
  return candidate[key] as T | undefined;
}
