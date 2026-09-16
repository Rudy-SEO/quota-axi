import {
  deleteCachedProvider as deleteCachedProviderFromDisk,
  readCachedProvider as readCachedProviderFromDisk,
} from "../cache.js";
import { readJsonFileResult, type JsonFileReadResult } from "../lib/fs.js";
import { providerFetch } from "../lib/http.js";
import { resolvePiAuthFilePath } from "../lib/pi-agent-dir.js";
import { classifyPiAuthEntry } from "../lib/pi-auth-store.js";
import { usableLiteralSecret } from "../lib/secret.js";
import type {
  AuthProviderReport,
  ProviderAdapter,
  ProviderOptions,
  ProviderQuota,
  ProviderStatus,
  QuotaWindow,
  SourceAttempt,
} from "../types.js";
import { VERSION } from "../version.js";

const OPENROUTER_HOST = "openrouter.ai";
const OPENROUTER_KEY_PATH = "/api/v1/auth/key";
const PI_OPENROUTER_SOURCE = "pi:openrouter";
const PI_OPENROUTER_PROVIDER_ID = "openrouter";
const LABEL = "OpenRouter";
const OPERATION_DEADLINE_MS = 15_000;
const RESPONSE_LIMIT_BYTES = 262_144;
const MONTH_SECONDS = 30 * 24 * 60 * 60;
const USER_AGENT = `quota-axi/${VERSION}`;

export const OPENROUTER_LIMIT_WINDOW_ID = "limit";
export const OPENROUTER_USAGE_WINDOW_IDS = [
  "usage_daily",
  "usage_weekly",
  "usage_monthly",
] as const;

export type OpenRouterCredentialResolution =
  | { status: "available"; apiKey: string; path: string }
  | { status: "missing"; path: string }
  | { status: "invalid"; path: string; error: string }
  | { status: "error"; path: string; error: string };

type OpenRouterDependencies = {
  credential: () => OpenRouterCredentialResolution;
  fetch: typeof globalThis.fetch;
  readCachedProvider: typeof readCachedProviderFromDisk;
  deleteCachedProvider: typeof deleteCachedProviderFromDisk;
  now: () => number;
  deadlineMs: number;
};

type OpenRouterFailureOptions = {
  status?: ProviderStatus;
  staleEligible?: boolean;
  definitiveAuth?: boolean;
  retryAfter?: string;
};

type ResponseBodyLifetime = {
  markConsumed(): void;
  cancel(action?: () => Promise<unknown> | undefined): Promise<void>;
};

export function resolveOpenRouterCredential(
  filePath: () => string = resolvePiAuthFilePath,
): OpenRouterCredentialResolution {
  const path = filePath();
  const result: JsonFileReadResult = readJsonFileResult(path);
  if (result.status === "missing") return { status: "missing", path };
  if (result.status === "invalid")
    return result.error === "file_read_error"
      ? { status: "error", path, error: result.error }
      : { status: "invalid", path, error: result.error };
  const classified = classifyPiAuthEntry(
    result.value,
    PI_OPENROUTER_PROVIDER_ID,
  );
  if (classified.status === "missing") return { status: "missing", path };
  const key =
    classified.status === "present" && classified.entry.type === "api_key"
      ? usableLiteralSecret(classified.entry.key)
      : undefined;
  return key
    ? { status: "available", apiKey: key, path }
    : { status: "invalid", path, error: "invalid_credential" };
}

export function createOpenRouterAdapter(
  overrides: Partial<OpenRouterDependencies> = {},
): ProviderAdapter {
  const dependencies: OpenRouterDependencies = {
    credential: () => resolveOpenRouterCredential(),
    fetch: providerFetch,
    readCachedProvider: readCachedProviderFromDisk,
    deleteCachedProvider: deleteCachedProviderFromDisk,
    now: Date.now,
    deadlineMs: OPERATION_DEADLINE_MS,
    ...overrides,
  };
  let inFlight: Promise<ProviderQuota> | undefined;

  return {
    id: "openrouter",
    label: LABEL,
    fetchQuota(_options: ProviderOptions): Promise<ProviderQuota> {
      if (inFlight) return inFlight;
      const acquisition = acquireOpenRouterQuota(dependencies).finally(() => {
        if (inFlight === acquisition) inFlight = undefined;
      });
      inFlight = acquisition;
      return acquisition;
    },
    async inspectAuth(_options: ProviderOptions): Promise<AuthProviderReport> {
      const resolution = dependencies.credential();
      return {
        provider: "openrouter",
        sources: [
          {
            source: PI_OPENROUTER_SOURCE,
            path: resolution.path,
            status: resolution.status,
            ...(resolution.status === "invalid" || resolution.status === "error"
              ? { error: resolution.error }
              : {}),
          },
        ],
      };
    },
  };
}

export const openrouterAdapter = createOpenRouterAdapter();

async function acquireOpenRouterQuota(
  dependencies: OpenRouterDependencies,
): Promise<ProviderQuota> {
  const controller = new AbortController();
  const deadline = setTimeout(
    () => controller.abort(),
    dependencies.deadlineMs,
  );
  const attempts: SourceAttempt[] = [];

  try {
    const resolution = dependencies.credential();
    if (resolution.status !== "available") {
      const failure = credentialFailureFor(resolution);
      attempts.push({
        source: PI_OPENROUTER_SOURCE,
        status: resolution.status === "missing" ? "skipped" : "failed",
        error: failure.code,
      });
      return failureReport(failure, attempts, dependencies);
    }

    attempts.push({ source: PI_OPENROUTER_SOURCE, status: "failed" });
    const payload = await requestOpenRouterKey(
      resolution.apiKey,
      controller.signal,
      dependencies.fetch,
      dependencies.now,
    );
    const receivedAt = dependencies.now();
    const windows = normalizeOpenRouterPayload(payload, receivedAt);
    const refreshedAt = new Date(receivedAt).toISOString();
    attempts[attempts.length - 1] = {
      source: PI_OPENROUTER_SOURCE,
      status: "success",
    };
    return {
      provider: "openrouter",
      label: LABEL,
      source: "api",
      windows,
      state: {
        status: "fresh",
        stale: false,
        refreshedAt,
        sourcesTried: attempts.map(({ source }) => source),
      },
      attempts,
    };
  } catch (error) {
    const failure =
      error instanceof OpenRouterFailure
        ? error
        : new OpenRouterFailure("credential_resolution_failed", {
            staleEligible: true,
          });
    if (attempts.length === 0) {
      attempts.push({
        source: PI_OPENROUTER_SOURCE,
        status: "failed",
        error: failure.code,
      });
    } else {
      attempts[attempts.length - 1] = {
        source: PI_OPENROUTER_SOURCE,
        status: "failed",
        error: failure.code,
      };
    }
    return failureReport(failure, attempts, dependencies);
  } finally {
    clearTimeout(deadline);
  }
}

function credentialFailureFor(
  resolution: Exclude<OpenRouterCredentialResolution, { status: "available" }>,
): OpenRouterFailure {
  if (resolution.status === "missing") {
    return new OpenRouterFailure("openrouter_credential_unavailable", {
      status: "auth_required",
      definitiveAuth: true,
    });
  }
  if (resolution.status === "error") {
    return new OpenRouterFailure("credential_resolution_failed", {
      staleEligible: true,
    });
  }
  return new OpenRouterFailure("openrouter_credential_invalid", {
    status: "auth_required",
    definitiveAuth: true,
  });
}

function failureReport(
  failure: OpenRouterFailure,
  attempts: SourceAttempt[],
  dependencies: OpenRouterDependencies,
): ProviderQuota {
  if (failure.definitiveAuth) {
    try {
      dependencies.deleteCachedProvider("openrouter");
    } catch {
      // The current auth failure is still definitive even if the cache is not writable.
    }
  }

  if (failure.staleEligible) {
    try {
      const cached = dependencies.readCachedProvider("openrouter");
      const stale = cached
        ? staleOpenRouterReport(
            cached,
            failure.code,
            failure.retryAfter,
            attempts,
            dependencies.now(),
          )
        : undefined;
      if (stale) return stale;
    } catch {
      // Cache I/O cannot replace the bounded current provider failure.
    }
  }

  return {
    provider: "openrouter",
    label: LABEL,
    source: "unavailable",
    windows: [],
    state: {
      status: failure.status,
      stale: false,
      error: failure.code,
      ...(failure.retryAfter ? { retryAfter: failure.retryAfter } : {}),
      sourcesTried: attempts.map(({ source }) => source),
    },
    attempts,
  };
}

function staleOpenRouterReport(
  cached: ProviderQuota,
  error: string,
  retryAfter: string | undefined,
  attempts: SourceAttempt[],
  now: number,
): ProviderQuota | undefined {
  if (
    cached.provider !== "openrouter" ||
    cached.source !== "api" ||
    cached.state.status !== "fresh" ||
    !cached.state.refreshedAt
  ) {
    return undefined;
  }
  const refreshedAt = Date.parse(cached.state.refreshedAt);
  if (!Number.isFinite(refreshedAt)) return undefined;
  const ageMilliseconds = Math.max(0, now - refreshedAt);
  const windows = cached.windows.filter((window) => {
    if (window.resetsAt !== undefined) {
      const resetsAt = Date.parse(window.resetsAt);
      return Number.isFinite(resetsAt) && resetsAt > now;
    }
    // A limit without a recognized reset cadence is a plain credit cap; a
    // month is the longest cycle this endpoint describes.
    return (
      window.id === OPENROUTER_LIMIT_WINDOW_ID &&
      ageMilliseconds < MONTH_SECONDS * 1_000
    );
  });
  if (windows.length === 0) return undefined;

  return {
    provider: "openrouter",
    label: LABEL,
    source: "cache",
    windows,
    state: {
      status: "stale",
      stale: true,
      refreshedAt: cached.state.refreshedAt,
      error,
      ...(retryAfter ? { retryAfter } : {}),
      sourcesTried: [...attempts.map(({ source }) => source), "cache"],
    },
    attempts,
  };
}

async function requestOpenRouterKey(
  apiKey: string,
  signal: AbortSignal,
  fetchImplementation: typeof globalThis.fetch,
  now: () => number,
): Promise<unknown> {
  let response: Response;
  try {
    response = await waitForDeadline(
      fetchImplementation(`https://${OPENROUTER_HOST}${OPENROUTER_KEY_PATH}`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          Accept: "application/json",
          "User-Agent": USER_AGENT,
        },
        credentials: "omit",
        redirect: "manual",
        signal,
      }),
      signal,
    );
  } catch (error) {
    if (signal.aborted || isAbortError(error)) {
      throw new OpenRouterFailure("request_timeout", { staleEligible: true });
    }
    throw new OpenRouterFailure(localTransportCode(error), {
      staleEligible: true,
    });
  }

  const lifetime = createResponseBodyLifetime(response);
  try {
    const receivedAt = now();
    rejectHttpFailure(response, receivedAt);

    let bytes: Uint8Array;
    try {
      bytes = await readBoundedBody(response, signal, lifetime);
      lifetime.markConsumed();
    } catch (error) {
      if (error instanceof OpenRouterFailure) throw error;
      if (signal.aborted || isAbortError(error)) {
        throw new OpenRouterFailure("request_timeout", { staleEligible: true });
      }
      throw new OpenRouterFailure("network_unavailable", {
        staleEligible: true,
      });
    }

    let text: string;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw new OpenRouterFailure("response_invalid_utf8");
    }
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw new OpenRouterFailure("malformed_json");
    }
  } finally {
    await lifetime.cancel();
  }
}

function rejectHttpFailure(response: Response, receivedAt: number): void {
  const status = response.status;
  if (status === 200) return;
  if (status >= 300 && status <= 399) {
    throw new OpenRouterFailure("redirect_rejected");
  }
  if (status === 401 || status === 403) {
    throw new OpenRouterFailure("provider_auth_rejected", {
      status: "auth_required",
      definitiveAuth: true,
    });
  }
  if (status === 408) {
    throw new OpenRouterFailure("provider_timeout", { staleEligible: true });
  }
  if (status === 429) {
    throw new OpenRouterFailure("provider_rate_limited", {
      status: "rate_limited",
      staleEligible: true,
      retryAfter: normalizeRetryAfter(
        response.headers.get("retry-after"),
        receivedAt,
      ),
    });
  }
  if (status >= 500 && status <= 599) {
    throw new OpenRouterFailure("provider_unavailable", {
      staleEligible: true,
    });
  }
  throw new OpenRouterFailure("provider_request_rejected");
}

async function readBoundedBody(
  response: Response,
  signal: AbortSignal,
  lifetime: ResponseBodyLifetime,
): Promise<Uint8Array> {
  const declaredLength = response.headers.get("content-length")?.trim();
  if (declaredLength && /^\d+$/.test(declaredLength)) {
    if (BigInt(declaredLength) > BigInt(RESPONSE_LIMIT_BYTES)) {
      throw new OpenRouterFailure("response_too_large", {
        staleEligible: true,
      });
    }
  }
  if (!response.body) return new Uint8Array();

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await readBodyChunk(reader, signal, lifetime);
      if (done) break;
      length += value.length;
      if (length > RESPONSE_LIMIT_BYTES) {
        throw new OpenRouterFailure("response_too_large", {
          staleEligible: true,
        });
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function readBodyChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal,
  lifetime: ResponseBodyLifetime,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  const cancelReader = () => lifetime.cancel(() => reader.cancel());
  if (signal.aborted) {
    await cancelReader();
    throw new OpenRouterFailure("request_timeout", { staleEligible: true });
  }
  return new Promise((resolve, reject) => {
    let aborted = false;
    const abort = () => {
      aborted = true;
      cancelReader().then(() => {
        reject(
          new OpenRouterFailure("request_timeout", { staleEligible: true }),
        );
      });
    };
    signal.addEventListener("abort", abort, { once: true });
    reader.read().then(
      (result) => {
        if (aborted) return;
        signal.removeEventListener("abort", abort);
        resolve(result);
      },
      (error: unknown) => {
        if (aborted) return;
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}

function createResponseBodyLifetime(response: Response): ResponseBodyLifetime {
  let consumed = false;
  let cancellation: Promise<void> | undefined;

  return {
    markConsumed() {
      if (!cancellation) consumed = true;
    },
    async cancel(action = () => response.body?.cancel()) {
      if (consumed) return;
      cancellation ??= Promise.resolve()
        .then(action)
        .then(() => undefined)
        .catch(() => undefined);
      await cancellation;
    },
  };
}

/**
 * Map the OpenRouter key-status record to quota windows.
 *
 * The key's `limit` / `limit_remaining` pair is the one enforced bound: a key
 * whose remaining limit reaches zero is refused. `limit_reset` names the
 * replenishment cadence as prose (`daily`, `weekly`, `monthly`) with no reset
 * timestamp; OpenRouter documents that limits reset at midnight UTC with
 * Monday-to-Sunday weeks, so a recognized cadence resolves to its current UTC
 * period. The `usage_*` fields are spend meters over the same UTC day, week,
 * and month with no cap of their own, reported as windows without percentages.
 */
export function normalizeOpenRouterPayload(
  payload: unknown,
  now: number,
): QuotaWindow[] {
  const root = objectValue(payload);
  const data = objectValue(root?.data) ?? root;
  if (!data || !isKeyRecord(data)) {
    throw new OpenRouterFailure("schema_invalid");
  }

  const windows: QuotaWindow[] = [];
  const limit = numericScalar(data.limit);
  if (limit !== undefined) {
    windows.push(limitWindow(limit, data, now));
  }
  for (const id of OPENROUTER_USAGE_WINDOW_IDS) {
    const spent = numericScalar(data[id]);
    if (spent === undefined || spent < 0) continue;
    windows.push({
      id,
      label: USAGE_WINDOWS[id].label,
      kind: "credits",
      spentUsd: spent,
      ...utcPeriod(USAGE_WINDOWS[id].period, now),
    });
  }
  return windows;
}

type UtcPeriod = "day" | "week" | "month";

const USAGE_WINDOWS: Record<
  (typeof OPENROUTER_USAGE_WINDOW_IDS)[number],
  { label: string; period: UtcPeriod }
> = {
  usage_daily: { label: "day usage", period: "day" },
  usage_weekly: { label: "week usage", period: "week" },
  usage_monthly: { label: "month usage", period: "month" },
};

const LIMIT_RESET_CADENCES: Record<
  string,
  { label: string; period: UtcPeriod; resetText: string }
> = {
  daily: { label: "day", period: "day", resetText: "daily" },
  weekly: { label: "week", period: "week", resetText: "weekly" },
  monthly: { label: "month", period: "month", resetText: "monthly" },
};

function utcPeriod(
  period: UtcPeriod,
  now: number,
): Pick<QuotaWindow, "startsAt" | "resetsAt"> {
  const date = new Date(now);
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();
  const day = date.getUTCDate();
  const mondayOffset = (date.getUTCDay() + 6) % 7;
  const [startsAt, resetsAt] =
    period === "day"
      ? [Date.UTC(year, month, day), Date.UTC(year, month, day + 1)]
      : period === "week"
        ? [
            Date.UTC(year, month, day - mondayOffset),
            Date.UTC(year, month, day - mondayOffset + 7),
          ]
        : [Date.UTC(year, month, 1), Date.UTC(year, month + 1, 1)];
  return {
    startsAt: new Date(startsAt).toISOString(),
    resetsAt: new Date(resetsAt).toISOString(),
  };
}

function limitWindow(
  limit: number,
  data: Record<string, unknown>,
  now: number,
): QuotaWindow {
  const remaining = numericScalar(data.limit_remaining);
  const cadenceValue = stringValue(data.limit_reset)?.toLowerCase();
  const cadence =
    cadenceValue && Object.hasOwn(LIMIT_RESET_CADENCES, cadenceValue)
      ? LIMIT_RESET_CADENCES[cadenceValue]
      : undefined;
  const percentRemaining =
    limit > 0 && remaining !== undefined
      ? clampPercent((remaining / limit) * 100)
      : limit === 0
        ? 0
        : undefined;
  return {
    id: OPENROUTER_LIMIT_WINDOW_ID,
    // A key limit without a recognized reset cadence is a plain credit cap.
    label: cadence?.label ?? "credits",
    kind: "credits",
    ...(percentRemaining !== undefined
      ? {
          percentRemaining,
          percentUsed: clampPercent(100 - percentRemaining),
        }
      : {}),
    ...(remaining !== undefined
      ? { spentUsd: Math.max(0, limit - remaining) }
      : {}),
    limitUsd: limit,
    ...(cadence
      ? { ...utcPeriod(cadence.period, now), resetText: cadence.resetText }
      : {}),
  };
}

const KEY_RECORD_FIELDS = [
  "limit",
  "limit_remaining",
  "limit_reset",
  "usage",
  "usage_daily",
  "usage_weekly",
  "usage_monthly",
];

function isKeyRecord(data: Record<string, unknown>): boolean {
  return KEY_RECORD_FIELDS.some((field) => Object.hasOwn(data, field));
}

function numericScalar(value: unknown): number | undefined {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  if (typeof value !== "string") return undefined;
  const trimmed = value.replace(/^[\t\n\v\f\r ]+|[\t\n\v\f\r ]+$/g, "");
  if (!/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    return undefined;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, value));
}

export function normalizeRetryAfter(
  value: string | null,
  receivedAt: number,
): string | undefined {
  const raw = value?.trim();
  if (!raw) return undefined;
  if (/^\d+$/.test(raw)) {
    const seconds = Number(raw);
    const instant = receivedAt + seconds * 1_000;
    if (!Number.isFinite(seconds) || !Number.isFinite(instant))
      return undefined;
    try {
      return new Date(instant).toISOString();
    } catch {
      return undefined;
    }
  }
  if (!/^[A-Za-z]/.test(raw)) return undefined;
  const parsed = Date.parse(raw);
  if (!Number.isFinite(parsed)) return undefined;
  try {
    return new Date(parsed).toISOString();
  } catch {
    return undefined;
  }
}

function localTransportCode(
  error: unknown,
): "tls_failed" | "network_unavailable" {
  const cause = objectValue(objectValue(error)?.cause);
  const code = typeof cause?.code === "string" ? cause.code : undefined;
  return code && /(?:TLS|SSL|CERT|UNABLE_TO_VERIFY)/i.test(code)
    ? "tls_failed"
    : "network_unavailable";
}

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function waitForDeadline<T>(
  promise: Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(
      new OpenRouterFailure("request_timeout", { staleEligible: true }),
    );
  }
  return new Promise<T>((resolve, reject) => {
    const abort = () =>
      reject(new OpenRouterFailure("request_timeout", { staleEligible: true }));
    signal.addEventListener("abort", abort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}

class OpenRouterFailure extends Error {
  readonly code: string;
  readonly status: ProviderStatus;
  readonly staleEligible: boolean;
  readonly definitiveAuth: boolean;
  readonly retryAfter?: string;

  constructor(code: string, options: OpenRouterFailureOptions = {}) {
    super(code);
    this.code = code;
    this.status = options.status ?? "error";
    this.staleEligible = options.staleEligible ?? false;
    this.definitiveAuth = options.definitiveAuth ?? false;
    this.retryAfter = options.retryAfter;
  }
}
