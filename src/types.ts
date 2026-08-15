export interface ToolCall {
  toolName: string;
  arguments: Record<string, unknown>;
  latencyMs: number;
}

export interface Trace {
  initialUserPrompt: string;
  trajectory: ToolCall[];
  totalTokensConsumed: number;
  finalOutput: string;
}
