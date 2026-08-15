# pox

Agent evaluation and trajectory testing framework for tool-using agents.

`pox` helps you:
- run prompt-based agent tests without live network calls
- capture a deterministic tool-call trajectory
- enforce regressions with golden traces
- add semantic output checks with an LLM judge

## Installation

### As a package

```bash
npm install pox
```

### Run with npx

After publish, the package exposes a binary so you can run:

```bash
npx pox
```

## CLI

Run all tests matching `*.pox.test.ts` recursively from the current directory:

```bash
npx pox
```

Useful options:

```bash
npx pox --dir ./tests --trace-file .pox_trace.json
```

Behavior:
- first run creates a baseline trajectory per test in `.pox_trace.json`
- later runs compare the new trajectory to baseline
- missing or extra tool calls are shown as a visual diff
- any divergence exits with status code `1` for CI/CD

## First Test

Create a file like `user-flow.pox.test.ts`:

```ts
import { expect, mockMCPTool, runPrompt, type Trace } from "pox";

class DemoAgent {
  private readonly transport: { send: (message: unknown) => Promise<void> };

  public constructor(options: { transport: { send: (message: unknown) => Promise<void> } }) {
    this.transport = options.transport;
  }

  public async run(input: string): Promise<{ finalOutput: string; totalTokensConsumed: number }> {
    await this.transport.send({
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: {
        name: "create_user",
        arguments: {
          userId: "u-1",
          name: "Ada",
        },
      },
    });

    await this.transport.send({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: {
        name: "get_user",
        arguments: {
          userId: "u-1",
        },
      },
    });

    return {
      finalOutput: `Created and verified user for prompt: ${input}`,
      totalTokensConsumed: 42,
    };
  }
}

export async function run(): Promise<Trace> {
  const AgentUnderTest = Object.assign(DemoAgent, {
    mockResponses: {
      ...mockMCPTool("create_user", (args, state) => ({
        ok: true,
        stored: {
          id: args.userId,
          name: args.name,
          totalUsers: Object.keys(state.users).length + 1,
        },
      })),
      ...mockMCPTool("get_user", (_args, state) => ({
        ok: true,
        usersInDb: Object.keys(state.users).length,
      })),
    },
  });

  const trace = await runPrompt(AgentUnderTest, "Create Ada and confirm retrieval");

  expect(trace)
    .toHaveCalled("create_user")
    .toHaveCalledBefore("create_user", "get_user");

  return trace;
}
```

Run it:

```bash
npx pox --dir .
```

## Assertion API

`expect(trace)` currently supports:
- `toHaveCalled(toolName)`
- `toHaveCalledBefore(toolA, toolB)`
- `await toSemanticallyMatch(expectedMeaning, judgeModel)`

Semantic matcher model formats:
- local Ollama: `ollama:llama3.1`
- OpenAI-compatible API: `openai:o3-mini`
- Groq: `groq:llama-3.1-8b-instant`
- Anthropic: `anthropic:claude-3-5-haiku-latest`

Environment variables:
- `OLLAMA_BASE_URL` (optional, default `http://localhost:11434/api/generate`)
- `OPENAI_API_KEY` (required for `openai:*` models)
- `OPENAI_BASE_URL` (optional override)
- `GROQ_API_KEY` (required for `groq:*` models)
- `GROQ_BASE_URL` (optional, default `https://api.groq.com/openai/v1/chat/completions`)
- `ANTHROPIC_API_KEY` (required for `anthropic:*` models)
- `ANTHROPIC_BASE_URL` (optional, default `https://api.anthropic.com/v1/messages`)
- `ANTHROPIC_VERSION` (optional, default `2023-06-01`)

## Project Scripts

```bash
npm run typecheck
npm run build
npm run pox
```

## Publishing Notes

The package is configured with:
- `bin.pox -> dist/cli.js` for `npx pox`
- `tsup` build for minified ESM bundles in `dist/`
- `prepublishOnly` to enforce typecheck + build before publish
