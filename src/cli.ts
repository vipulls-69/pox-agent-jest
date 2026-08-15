import "dotenv/config";

import { access, readFile, writeFile } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { Command } from "commander";

import type { ToolCall, Trace } from "./types.js";

interface TraceEnvelope {
  name: string;
  trace: Trace;
}

interface StoredTrajectory {
  trajectory: ToolCall[];
}

interface TraceStore {
  version: 1;
  tests: Record<string, StoredTrajectory>;
}

type RunnerResult = Trace | TraceEnvelope | Array<Trace | TraceEnvelope>;
type TestRunner = () => RunnerResult | Promise<RunnerResult>;

type DiffOp =
  | { kind: "equal"; value: string }
  | { kind: "remove"; value: string }
  | { kind: "add"; value: string };

const COLORS = {
  reset: "\x1b[0m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  cyan: "\x1b[36m",
  yellow: "\x1b[33m",
};

const program = new Command();

program
  .name("pox")
  .description("Run *.pox.test.ts files with trajectory regression tracking")
  .option("-d, --dir <path>", "Directory to scan for test files", process.cwd())
  .option("-t, --trace-file <name>", "Golden trace file name", ".pox_trace.json")
  .action(async (options: { dir: string; traceFile: string }) => {
    await runCli(options.dir, options.traceFile);
  });

void program.parseAsync(process.argv);

export async function runCli(rootDir: string, traceFileName: string): Promise<void> {
  const scanRoot = resolve(rootDir);
  const traceFilePath = join(scanRoot, traceFileName);

  const testFiles = await findTestFiles(scanRoot);
  if (testFiles.length === 0) {
    console.log(`${COLORS.yellow}No *.pox.test.ts files found in ${scanRoot}${COLORS.reset}`);
    return;
  }

  const store = await loadTraceStore(traceFilePath);
  let baselineUpdated = false;
  let regressionFound = false;

  for (const testFile of testFiles) {
    const rel = relative(scanRoot, testFile);
    console.log(`${COLORS.cyan}Running ${rel}${COLORS.reset}`);

    let envelopes: TraceEnvelope[];
    try {
      envelopes = await executeTestFile(testFile, rel);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`${COLORS.red}Test execution failed for ${rel}: ${message}${COLORS.reset}`);
      process.exit(1);
      return;
    }

    for (const envelope of envelopes) {
      const testId = `${rel}::${envelope.name}`;
      const existing = store.tests[testId];

      if (!existing) {
        store.tests[testId] = { trajectory: envelope.trace.trajectory };
        baselineUpdated = true;
        console.log(`${COLORS.green}  Created baseline for ${testId}${COLORS.reset}`);
        continue;
      }

      const expected = existing.trajectory.map(formatToolCall);
      const actual = envelope.trace.trajectory.map(formatToolCall);
      const same = sequencesEqual(expected, actual);

      if (!same) {
        regressionFound = true;
        console.error(`${COLORS.red}  Regression in ${testId}${COLORS.reset}`);
        printTrajectoryDiff(expected, actual);
      } else {
        console.log(`${COLORS.green}  Pass ${testId}${COLORS.reset}`);
      }
    }
  }

  if (baselineUpdated) {
    await writeFile(traceFilePath, JSON.stringify(store, null, 2), "utf8");
    console.log(`${COLORS.cyan}Saved baselines to ${traceFilePath}${COLORS.reset}`);
  }

  if (regressionFound) {
    process.exit(1);
  }
}

async function executeTestFile(filePath: string, relativePath: string): Promise<TraceEnvelope[]> {
  const moduleUrl = pathToFileURL(filePath).href;
  const mod = (await import(moduleUrl)) as Record<string, unknown>;

  const runner = resolveRunner(mod);
  if (!runner) {
    throw new Error(`No runnable export found in ${relativePath}. Export run() or default function.`);
  }

  const result = await runner();
  return normalizeRunnerResult(result, relativePath);
}

function resolveRunner(mod: Record<string, unknown>): TestRunner | undefined {
  const runCandidate = mod.run;
  if (typeof runCandidate === "function") {
    return runCandidate as TestRunner;
  }

  const defaultCandidate = mod.default;
  if (typeof defaultCandidate === "function") {
    return defaultCandidate as TestRunner;
  }

  return undefined;
}

function normalizeRunnerResult(result: RunnerResult, fallbackName: string): TraceEnvelope[] {
  const items = Array.isArray(result) ? result : [result];
  const envelopes: TraceEnvelope[] = [];

  for (const item of items) {
    if (isTrace(item)) {
      envelopes.push({ name: fallbackName, trace: item });
      continue;
    }

    if (isTraceEnvelope(item)) {
      envelopes.push({
        name: item.name,
        trace: item.trace,
      });
      continue;
    }

    throw new Error("Runner returned an invalid result. Expected Trace or { name, trace }.");
  }

  return envelopes;
}

async function findTestFiles(rootDir: string): Promise<string[]> {
  const output: string[] = [];

  async function walk(currentDir: string): Promise<void> {
    const entries = await readDirSafe(currentDir);

    for (const entry of entries) {
      const fullPath = join(currentDir, entry.name);

      if (entry.isDirectory()) {
        if (entry.name === "node_modules" || entry.name === ".git" || entry.name === "dist") {
          continue;
        }

        await walk(fullPath);
        continue;
      }

      if (entry.isFile() && entry.name.endsWith(".pox.test.ts")) {
        output.push(fullPath);
      }
    }
  }

  await walk(rootDir);
  output.sort();
  return output;
}

async function readDirSafe(dirPath: string): Promise<Array<import("node:fs").Dirent>> {
  const { readdir } = await import("node:fs/promises");
  return readdir(dirPath, { withFileTypes: true });
}

async function loadTraceStore(filePath: string): Promise<TraceStore> {
  if (!(await fileExists(filePath))) {
    return {
      version: 1,
      tests: {},
    };
  }

  const raw = await readFile(filePath, "utf8");
  const parsed = JSON.parse(raw) as Partial<TraceStore>;

  if (!parsed || parsed.version !== 1 || typeof parsed.tests !== "object" || !parsed.tests) {
    throw new Error(`Invalid trace store format in ${filePath}`);
  }

  return {
    version: 1,
    tests: parsed.tests as Record<string, StoredTrajectory>,
  };
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function sequencesEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) {
    return false;
  }

  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }

  return true;
}

function printTrajectoryDiff(expected: string[], actual: string[]): void {
  const ops = buildDiff(expected, actual);
  console.error("  Trajectory diff:");

  for (const op of ops) {
    if (op.kind === "equal") {
      console.error(`    ${op.value}`);
      continue;
    }

    if (op.kind === "remove") {
      console.error(`    ${COLORS.red}- ${op.value}${COLORS.reset}`);
      continue;
    }

    console.error(`    ${COLORS.green}+ ${op.value}${COLORS.reset}`);
  }
}

function buildDiff(expected: string[], actual: string[]): DiffOp[] {
  const n = expected.length;
  const m = actual.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => Array<number>(m + 1).fill(0));

  for (let i = n - 1; i >= 0; i -= 1) {
    const dpRow = dp[i];
    const nextRow = dp[i + 1];
    if (!dpRow || !nextRow) {
      continue;
    }

    for (let j = m - 1; j >= 0; j -= 1) {
      const expectedValue = expected[i];
      const actualValue = actual[j];
      const nextDiag = nextRow[j + 1] ?? 0;
      const nextDown = nextRow[j] ?? 0;
      const nextRight = dpRow[j + 1] ?? 0;

      if (expectedValue === actualValue) {
        dpRow[j] = nextDiag + 1;
      } else {
        dpRow[j] = Math.max(nextDown, nextRight);
      }
    }
  }

  const ops: DiffOp[] = [];
  let i = 0;
  let j = 0;

  while (i < n && j < m) {
    const expectedValue = expected[i];
    const actualValue = actual[j];
    if (expectedValue === undefined || actualValue === undefined) {
      break;
    }

    if (expectedValue === actualValue) {
      ops.push({ kind: "equal", value: expectedValue });
      i += 1;
      j += 1;
      continue;
    }

    const downScore = dp[i + 1]?.[j] ?? 0;
    const rightScore = dp[i]?.[j + 1] ?? 0;
    if (downScore >= rightScore) {
      ops.push({ kind: "remove", value: expectedValue });
      i += 1;
      continue;
    }

    ops.push({ kind: "add", value: actualValue });
    j += 1;
  }

  while (i < n) {
    const expectedValue = expected[i];
    if (expectedValue !== undefined) {
      ops.push({ kind: "remove", value: expectedValue });
    }
    i += 1;
  }

  while (j < m) {
    const actualValue = actual[j];
    if (actualValue !== undefined) {
      ops.push({ kind: "add", value: actualValue });
    }
    j += 1;
  }

  return ops;
}

function formatToolCall(call: ToolCall): string {
  const args = stableStringify(call.arguments);
  if (args === "{}") {
    return call.toolName;
  }

  return `${call.toolName} ${args}`;
}

function stableStringify(value: unknown): string {
  return JSON.stringify(sortUnknown(value));
}

function sortUnknown(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => sortUnknown(item));
  }

  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const sorted: Record<string, unknown> = {};

    for (const key of keys) {
      sorted[key] = sortUnknown(obj[key]);
    }

    return sorted;
  }

  return value;
}

function isTrace(value: unknown): value is Trace {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<Trace>;
  return (
    typeof candidate.initialUserPrompt === "string" &&
    Array.isArray(candidate.trajectory) &&
    typeof candidate.totalTokensConsumed === "number" &&
    typeof candidate.finalOutput === "string"
  );
}

function isTraceEnvelope(value: unknown): value is TraceEnvelope {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as { name?: unknown; trace?: unknown };
  return typeof candidate.name === "string" && isTrace(candidate.trace);
}
