import { randomUUID } from "node:crypto";

import type { Transport, TransportSendOptions } from "@modelcontextprotocol/sdk/shared/transport.js";
import {
  ErrorCode,
  isJSONRPCNotification,
  isJSONRPCRequest,
  JSONRPC_VERSION,
  type JSONRPCMessage,
  type JSONRPCRequest,
  type RequestId,
} from "@modelcontextprotocol/sdk/types.js";

export type MockResponseResolver =
  | JSONRPCMessage
  | ((request: JSONRPCRequest, state: MockState) => JSONRPCMessage | Record<string, unknown> | undefined);

type ToolMockResolver =
  | JSONRPCMessage
  | Record<string, unknown>
  | ((args: Record<string, unknown>, state: MockState, request: JSONRPCRequest) => JSONRPCMessage | Record<string, unknown> | undefined);

export interface MockState {
  users: Record<string, Record<string, unknown>>;
}

export function mockMCPTool(toolName: string, resolver: ToolMockResolver): Record<string, MockResponseResolver> {
  return {
    [toolName]: (request: JSONRPCRequest, state: MockState) => {
      const params = toRecordUnsafe(request.params);
      const requestToolName = params && typeof params.name === "string" ? params.name : undefined;

      if (request.method !== "tools/call" || requestToolName !== toolName) {
        return undefined;
      }

      if (typeof resolver !== "function") {
        return resolver;
      }

      const args = params ? toRecordUnsafe(params.arguments) ?? {} : {};
      return resolver(args, state, request);
    },
  };
}

export class MockTransport implements Transport {
  public onclose?: () => void;
  public onerror?: (error: Error) => void;
  public onmessage?: <T extends JSONRPCMessage>(message: T) => void;
  public sessionId?: string;
  public setProtocolVersion?: (version: string) => void;

  private readonly messageQueue: JSONRPCMessage[] = [];
  private readonly mockResponses: Record<string, MockResponseResolver>;
  private started = false;
  private closed = false;
  private readonly state: MockState;

  public constructor(
    mockResponses: Record<string, MockResponseResolver> = {},
    initialState?: Partial<MockState>,
  ) {
    this.sessionId = randomUUID();
    this.mockResponses = mockResponses;
    this.state = {
      users: { ...(initialState?.users ?? {}) },
    };
  }

  public async start(): Promise<void> {
    this.started = true;
    this.closed = false;
  }

  public async close(): Promise<void> {
    this.closed = true;
    this.started = false;
    this.onclose?.();
  }

  public async send(message: JSONRPCMessage, _options?: TransportSendOptions): Promise<void> {
    if (!this.started) {
      const error = new Error("MockTransport has not been started. Call start() first.");
      this.onerror?.(error);
      throw error;
    }

    if (this.closed) {
      const error = new Error("MockTransport is closed.");
      this.onerror?.(error);
      throw error;
    }

    this.messageQueue.push(message);
    this.drainQueue();
  }

  public getQueuedMessages(): readonly JSONRPCMessage[] {
    return this.messageQueue;
  }

  public getState(): Readonly<MockState> {
    return {
      users: { ...this.state.users },
    };
  }

  private drainQueue(): void {
    while (this.messageQueue.length > 0) {
      const next = this.messageQueue.shift();
      if (!next) {
        continue;
      }

      this.handleIncoming(next);
    }
  }

  private handleIncoming(message: JSONRPCMessage): void {
    if (isJSONRPCRequest(message)) {
      const response = this.resolveRequest(message);
      if (response) {
        this.onmessage?.(response);
      }
      return;
    }

    if (isJSONRPCNotification(message)) {
      this.applyStateMutation(message.method, message.params);
    }
  }

  private resolveRequest(request: JSONRPCRequest): JSONRPCMessage | undefined {
    this.applyStateMutation(request.method, request.params);

    const mockKey = this.getMockResponseKey(request);
    const resolver = mockKey ? this.mockResponses[mockKey] : undefined;

    if (resolver) {
      return this.resolveWithMock(resolver, request);
    }

    if (request.method === "tools/call") {
      const toolName = this.getToolName(request);
      const args = this.getToolArguments(request);

      if (toolName === "create_user") {
        const userId = this.extractUserId(args);
        if (!userId) {
          return this.makeErrorResponse(request.id, ErrorCode.InvalidParams, "create_user requires userId or id.");
        }

        this.state.users[userId] = {
          ...(this.state.users[userId] ?? {}),
          ...args,
          id: userId,
        };

        return this.makeResultResponse(request.id, {
          tool: toolName,
          ok: true,
          user: this.state.users[userId],
        });
      }

      if (toolName === "get_user") {
        const userId = this.extractUserId(args);
        if (!userId) {
          return this.makeErrorResponse(request.id, ErrorCode.InvalidParams, "get_user requires userId or id.");
        }

        return this.makeResultResponse(request.id, {
          tool: toolName,
          ok: true,
          user: this.state.users[userId] ?? null,
        });
      }
    }

    return this.makeErrorResponse(request.id, ErrorCode.MethodNotFound, `No mock response found for method: ${request.method}`);
  }

  private resolveWithMock(resolver: MockResponseResolver, request: JSONRPCRequest): JSONRPCMessage {
    if (typeof resolver !== "function") {
      return resolver;
    }

    const resolved = resolver(request, this.state);
    if (!resolved) {
      return this.makeResultResponse(request.id, {});
    }

    if (this.isJSONRPCMessage(resolved)) {
      return resolved;
    }

    return this.makeResultResponse(request.id, resolved);
  }

  private getMockResponseKey(request: JSONRPCRequest): string | undefined {
    if (request.method === "tools/call") {
      const toolName = this.getToolName(request);
      if (toolName) {
        return toolName;
      }
    }

    return request.method;
  }

  private applyStateMutation(method: string, params: unknown): void {
    if (method !== "tools/call") {
      return;
    }

    const paramsObj = this.toRecord(params);
    if (!paramsObj) {
      return;
    }

    const toolName = typeof paramsObj.name === "string" ? paramsObj.name : undefined;
    if (toolName !== "create_user") {
      return;
    }

    const args = this.toRecord(paramsObj.arguments);
    if (!args) {
      return;
    }

    const userId = this.extractUserId(args);
    if (!userId) {
      return;
    }

    this.state.users[userId] = {
      ...(this.state.users[userId] ?? {}),
      ...args,
      id: userId,
    };
  }

  private getToolName(request: JSONRPCRequest): string | undefined {
    const params = this.toRecord(request.params);
    if (!params) {
      return undefined;
    }

    return typeof params.name === "string" ? params.name : undefined;
  }

  private getToolArguments(request: JSONRPCRequest): Record<string, unknown> {
    const params = this.toRecord(request.params);
    if (!params) {
      return {};
    }

    return this.toRecord(params.arguments) ?? {};
  }

  private extractUserId(args: Record<string, unknown>): string | undefined {
    const candidate = args.userId ?? args.id;
    return typeof candidate === "string" && candidate.length > 0 ? candidate : undefined;
  }

  private makeResultResponse(id: RequestId, result: Record<string, unknown>): JSONRPCMessage {
    return {
      jsonrpc: JSONRPC_VERSION,
      id,
      result,
    };
  }

  private makeErrorResponse(id: RequestId, code: number, message: string): JSONRPCMessage {
    return {
      jsonrpc: JSONRPC_VERSION,
      id,
      error: {
        code,
        message,
      },
    };
  }

  private isJSONRPCMessage(value: unknown): value is JSONRPCMessage {
    if (!value || typeof value !== "object") {
      return false;
    }

    const candidate = value as { jsonrpc?: unknown };
    return candidate.jsonrpc === JSONRPC_VERSION;
  }

  private toRecord(value: unknown): Record<string, unknown> | undefined {
    return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
  }
}

function toRecordUnsafe(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}
