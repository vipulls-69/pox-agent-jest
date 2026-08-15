import type { Trace } from "./types.js";

interface SemanticJudgeOutcome {
  pass: boolean;
  reason?: string;
  raw?: string;
}

interface OpenAIResponse {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
}

interface AnthropicResponse {
  content?: Array<{
    type?: string;
    text?: string;
  }>;
}

interface OllamaGenerateResponse {
  response?: string;
}

export class TraceAssertionError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "TraceAssertionError";
  }
}

export class TraceExpect {
  private readonly trace: Trace;

  public constructor(trace: Trace) {
    this.trace = trace;
  }

  public toHaveCalled(toolName: string): this {
    const found = this.trace.trajectory.some((call) => call.toolName === toolName);
    if (!found) {
      throw new TraceAssertionError(
        `Expected tool \"${toolName}\" to have been called, but it was not found in trajectory.`,
      );
    }

    return this;
  }

  public toHaveCalledBefore(toolA: string, toolB: string): this {
    let firstA = -1;
    let firstB = -1;

    for (const [i, call] of this.trace.trajectory.entries()) {
      if (firstA === -1 && call.toolName === toolA) {
        firstA = i;
      }

      if (firstB === -1 && call.toolName === toolB) {
        firstB = i;
      }

      if (firstA !== -1 && firstB !== -1) {
        break;
      }
    }

    if (firstA === -1) {
      throw new TraceAssertionError(`Expected tool \"${toolA}\" to appear in trajectory, but it was never called.`);
    }

    if (firstB === -1) {
      throw new TraceAssertionError(`Expected tool \"${toolB}\" to appear in trajectory, but it was never called.`);
    }

    if (firstA >= firstB) {
      throw new TraceAssertionError(
        `Expected tool \"${toolA}\" to be called before \"${toolB}\", but indices were ${firstA} and ${firstB}.`,
      );
    }

    return this;
  }

  public async toSemanticallyMatch(expectedMeaning: string, judgeModel: string): Promise<this> {
    const outcome = await evaluateSemanticMatch(this.trace.finalOutput, expectedMeaning, judgeModel);

    if (!outcome.pass) {
      const suffix = outcome.reason ? ` Reason: ${outcome.reason}` : "";
      throw new TraceAssertionError(
        `Expected final output to semantically match \"${expectedMeaning}\", but judge returned fail.${suffix}`,
      );
    }

    return this;
  }
}

export function expectTrace(trace: Trace): TraceExpect {
  return new TraceExpect(trace);
}

export function expect(trace: Trace): TraceExpect {
  return expectTrace(trace);
}

export async function evaluateSemanticMatch(
  actualOutput: string,
  expectedMeaning: string,
  judgeModel: string,
): Promise<SemanticJudgeOutcome> {
  if (!judgeModel.trim()) {
    throw new Error("judgeModel is required.");
  }

  const systemPrompt = [
    "You are a strict evaluator for agent outputs.",
    "Return ONLY minified JSON with keys: pass (boolean), reason (string).",
    "Set pass=true only if actual output fulfills the semantic meaning of expected output.",
    "Ignore superficial wording differences.",
    "If uncertain, set pass=false and explain concisely in reason.",
  ].join(" ");

  const userPrompt = [
    `Expected meaning: ${expectedMeaning}`,
    `Actual output: ${actualOutput}`,
    "Respond with JSON only.",
  ].join("\n");

  if (judgeModel.startsWith("ollama:")) {
    const modelName = judgeModel.slice("ollama:".length).trim();
    if (!modelName) {
      throw new Error("For Ollama, use judgeModel format: ollama:<model-name>.");
    }

    const responseText = await callOllamaJudge(modelName, systemPrompt, userPrompt);
    return parseJudgeOutput(responseText);
  }

  if (judgeModel.startsWith("openai:")) {
    const modelName = judgeModel.slice("openai:".length).trim();
    if (!modelName) {
      throw new Error("For OpenAI, use judgeModel format: openai:<model-name>.");
    }

    const responseText = await callOpenAIJudge(modelName, systemPrompt, userPrompt);
    return parseJudgeOutput(responseText);
  }

  if (judgeModel.startsWith("groq:")) {
    const modelName = judgeModel.slice("groq:".length).trim();
    if (!modelName) {
      throw new Error("For Groq, use judgeModel format: groq:<model-name>.");
    }

    const responseText = await callGroqJudge(modelName, systemPrompt, userPrompt);
    return parseJudgeOutput(responseText);
  }

  if (judgeModel.startsWith("anthropic:")) {
    const modelName = judgeModel.slice("anthropic:".length).trim();
    if (!modelName) {
      throw new Error("For Anthropic, use judgeModel format: anthropic:<model-name>.");
    }

    const responseText = await callAnthropicJudge(modelName, systemPrompt, userPrompt);
    return parseJudgeOutput(responseText);
  }

  throw new Error("Unsupported judgeModel. Use ollama:<model>, openai:<model>, groq:<model>, or anthropic:<model>.");
}

async function callOllamaJudge(model: string, systemPrompt: string, userPrompt: string): Promise<string> {
  const url = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/api/generate";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      prompt: `${systemPrompt}\n\n${userPrompt}`,
      stream: false,
      format: "json",
      options: {
        temperature: 0,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama judge request failed with status ${response.status}.`);
  }

  const payload = (await response.json()) as OllamaGenerateResponse;
  return payload.response ?? "";
}

async function callOpenAIJudge(model: string, systemPrompt: string, userPrompt: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required for openai:* judge models.");
  }

  const url = process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1/chat/completions";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      response_format: {
        type: "json_object",
      },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI judge request failed with status ${response.status}.`);
  }

  const payload = (await response.json()) as OpenAIResponse;
  const message = payload.choices?.[0]?.message?.content;
  return message ?? "";
}

async function callGroqJudge(model: string, systemPrompt: string, userPrompt: string): Promise<string> {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error("GROQ_API_KEY is required for groq:* judge models.");
  }

  const url = process.env.GROQ_BASE_URL ?? "https://api.groq.com/openai/v1/chat/completions";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      response_format: {
        type: "json_object",
      },
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`Groq judge request failed with status ${response.status}.`);
  }

  const payload = (await response.json()) as OpenAIResponse;
  const message = payload.choices?.[0]?.message?.content;
  return message ?? "";
}

async function callAnthropicJudge(model: string, systemPrompt: string, userPrompt: string): Promise<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is required for anthropic:* judge models.");
  }

  const url = process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com/v1/messages";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": process.env.ANTHROPIC_VERSION ?? "2023-06-01",
    },
    body: JSON.stringify({
      model,
      system: systemPrompt,
      max_tokens: 256,
      temperature: 0,
      messages: [
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(`Anthropic judge request failed with status ${response.status}.`);
  }

  const payload = (await response.json()) as AnthropicResponse;
  const text = payload.content?.find((block) => block.type === "text")?.text;
  return text ?? "";
}

function parseJudgeOutput(raw: string): SemanticJudgeOutcome {
  const cleaned = extractJsonObject(raw);
  let parsed: unknown;

  try {
    parsed = JSON.parse(cleaned);
  } catch {
    return {
      pass: false,
      reason: "Judge did not return valid JSON.",
      raw,
    };
  }

  if (!parsed || typeof parsed !== "object") {
    return {
      pass: false,
      reason: "Judge response JSON is not an object.",
      raw,
    };
  }

  const candidate = parsed as { pass?: unknown; reason?: unknown };
  if (typeof candidate.pass !== "boolean") {
    return {
      pass: false,
      reason: "Judge response is missing boolean pass.",
      raw,
    };
  }

  const outcome: SemanticJudgeOutcome = {
    pass: candidate.pass,
    raw,
  };

  if (typeof candidate.reason === "string") {
    outcome.reason = candidate.reason;
  }

  return outcome;
}

function extractJsonObject(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed;
  }

  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    return trimmed.slice(start, end + 1);
  }

  return trimmed;
}
