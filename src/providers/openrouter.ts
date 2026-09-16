import {
  deleteCachedOpenRouterProvider as deleteCachedProviderFromDisk,
  readCachedOpenRouterProvider as readCachedProviderFromDisk,
} from "../cache.js";
import {
  openrouterCredentialContextId,
  readJsonFileResult,
  type JsonFileReadResult,
} from "../lib/fs.js";
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
const BODY_CLEANUP_TIMEOUT_MS = 100;
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
  const attempts: SourceAttempt[] = [];
  let contextId: string | undefined;

  try {
    const resolution = dependencies.credential();
    contextId = openrouterCredentialContextId(resolution.path);
    if (resolution.status !== "available") {
      const failure = credentialFailureFor(resolution);
      attempts.push({
        source: PI_OPENROUTER_SOURCE,
        status: resolution.status === "missing" ? "skipped" : "failed",
        error: failure.code,
      });
      return failureReport(failure, attempts, contextId, dependencies);
    }

    attempts.push({ source: PI_OPENROUTER_SOURCE, status: "failed" });
    const payload = await requestOpenRouterKey(
      resolution.apiKey,
      dependencies.fetch,
      dependencies.deadlineMs,
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
    return failureReport(failure, attempts, contextId, dependencies);
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
  contextId: string | undefined,
  dependencies: OpenRouterDependencies,
): ProviderQuota {
  // Without the auth file the reading was scoped to, no snapshot belongs to it.
  if (failure.definitiveAuth && contextId) {
    try {
      dependencies.deleteCachedProvider(contextId);
    } catch {
      // The current auth failure is still definitive even if the cache is not writable.
    }
  }

  if (failure.staleEligible && contextId) {
    try {
      const cached = dependencies.readCachedProvider(contextId);
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
  fetchImplementation: typeof globalThis.fetch,
  deadlineMs: number,
  now: () => number,
): Promise<unknown> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  try {
    const fetchPromise = fetchImplementation(
      `https://${OPENROUTER_HOST}${OPENROUTER_KEY_PATH}`,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          Accept: "application/json",
          "User-Agent": USER_AGENT,
        },
        credentials: "omit",
        redirect: "manual",
        signal: controller.signal,
      },
    );
    void fetchPromise.then(
      (response) => {
        if (timedOut) void cancelResponseBody(response);
      },
      () => undefined,
    );
    // The deadline rejects on its own schedule, so no cleanup of a stalled
    // transport can hold the operation past it.
    const deadline = new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        timedOut = true;
        controller.abort();
        reject(timeoutFailure());
      }, deadlineMs);
    });
    const response = await Promise.race([fetchPromise, deadline]);
    if (response.status !== 200) {
      await cancelResponseBody(response);
      rejectHttpFailure(response, now());
    }

    const bytes = await Promise.race([
      readBoundedBody(response, controller.signal),
      deadline,
    ]);
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
  } catch (error) {
    if (error instanceof OpenRouterFailure) throw error;
    if (controller.signal.aborted || isAbortError(error)) {
      throw timeoutFailure();
    }
    throw new OpenRouterFailure(localTransportCode(error), {
      staleEligible: true,
    });
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function rejectHttpFailure(response: Response, receivedAt: number): never {
  const status = response.status;
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

/**
 * Cancels a response body without letting a stalled transport hold the caller
 * past a short cleanup bound.
 */
async function cancelResponseBody(response: Response): Promise<void> {
  const body = response.body;
  if (!body) return;
  await settleWithin(
    Promise.resolve()
      .then(() => body.cancel())
      .catch(() => undefined),
  );
}

/**
 * Reads the body while counting decoded bytes, so a small declared length
 * cannot admit an oversized payload.
 */
async function readBoundedBody(
  response: Response,
  signal: AbortSignal,
): Promise<Uint8Array> {
  const declaredLength = response.headers.get("content-length")?.trim();
  if (
    declaredLength &&
    /^\d+$/.test(declaredLength) &&
    BigInt(declaredLength) > BigInt(RESPONSE_LIMIT_BYTES)
  ) {
    await cancelResponseBody(response);
    throw new OpenRouterFailure("response_too_large", { staleEligible: true });
  }
  if (!response.body) return new Uint8Array();

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  let pendingRead: Promise<ReadableStreamReadResult<Uint8Array>> | undefined;
  try {
    while (true) {
      pendingRead = reader.read();
      const result = await raceWithAbort(pendingRead, signal);
      pendingRead = undefined;
      if (result.done) break;
      length += result.value.byteLength;
      if (length > RESPONSE_LIMIT_BYTES) {
        throw new OpenRouterFailure("response_too_large", {
          staleEligible: true,
        });
      }
      chunks.push(result.value);
    }
  } finally {
    if (pendingRead) {
      await settlePendingRead(reader, pendingRead);
    } else {
      // Cancel before releasing the lock: a released reader cannot cancel.
      void reader.cancel().catch(() => undefined);
      reader.releaseLock();
    }
  }

  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function settlePendingRead(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  pendingRead: Promise<ReadableStreamReadResult<Uint8Array>>,
): Promise<void> {
  // A pending read owns the stream lock, and releasing it before the read
  // settles throws. Cancellation is best effort; the lock is released whenever
  // the read eventually settles, even after this bounded cleanup returns.
  void Promise.resolve()
    .then(() => reader.cancel())
    .catch(() => undefined);
  const releaseLock = (): void => {
    try {
      reader.releaseLock();
    } catch {
      // The stream already released or errored the lock.
    }
  };
  await settleWithin(pendingRead.then(releaseLock, releaseLock));
}

async function settleWithin(operation: Promise<void>): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      operation,
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, BODY_CLEANUP_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function raceWithAbort<T>(
  operation: Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  if (signal.aborted) throw timeoutFailure();
  let onAbort: (() => void) | undefined;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(timeoutFailure());
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    return await Promise.race([operation, aborted]);
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
  }
}

function timeoutFailure(): OpenRouterFailure {
  return new OpenRouterFailure("request_timeout", { staleEligible: true });
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
 *
 * A record is accepted only when every recognized field has its documented
 * shape and `limit` is present: `null` there is an explicitly unlimited key,
 * while a missing or malformed field is an unusable response, never an empty
 * reading that would retire the last good snapshot.
 */
export function normalizeOpenRouterPayload(
  payload: unknown,
  now: number,
): QuotaWindow[] {
  const data = objectValue(objectValue(payload)?.data);
  if (!data || !Object.hasOwn(data, "limit")) throw schemaInvalid();
  const limit = nullableNumber(data.limit);
  const remaining = nullableNumber(data.limit_remaining);
  const cadenceValue = nullableString(data.limit_reset);
  spendMeter(data.usage);

  const windows: QuotaWindow[] = [];
  if (limit !== undefined) {
    if (limit < 0 || remaining === undefined) throw schemaInvalid();
    windows.push(limitWindow(limit, remaining, cadenceValue, now));
  }
  for (const id of OPENROUTER_USAGE_WINDOW_IDS) {
    const spent = spendMeter(data[id]);
    if (spent === undefined) continue;
    windows.push({
      id,
      label: USAGE_WINDOWS[id].label,
      kind: "credits",
      spentUsd: spent,
      ...utcPeriod(USAGE_WINDOWS[id].period, now),
    });
  }
  if (windows.length === 0) throw schemaInvalid();
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
  remaining: number,
  cadenceValue: string | undefined,
  now: number,
): QuotaWindow {
  const cadenceKey = cadenceValue?.toLowerCase();
  const cadence =
    cadenceKey && Object.hasOwn(LIMIT_RESET_CADENCES, cadenceKey)
      ? LIMIT_RESET_CADENCES[cadenceKey]
      : undefined;
  const percentRemaining =
    limit > 0 ? clampPercent((remaining / limit) * 100) : 0;
  return {
    id: OPENROUTER_LIMIT_WINDOW_ID,
    // A key limit without a recognized reset cadence is a plain credit cap.
    label: cadence?.label ?? "credits",
    kind: "credits",
    percentRemaining,
    percentUsed: clampPercent(100 - percentRemaining),
    spentUsd: Math.max(0, limit - remaining),
    limitUsd: limit,
    ...(cadence
      ? { ...utcPeriod(cadence.period, now), resetText: cadence.resetText }
      : {}),
  };
}

function schemaInvalid(): OpenRouterFailure {
  return new OpenRouterFailure("schema_invalid");
}

/** A nullable dollar amount: absent or `null` is unreported, anything else must be numeric. */
function nullableNumber(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined;
  const number = numericScalar(value);
  if (number === undefined) throw schemaInvalid();
  return number;
}

function nullableString(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string") throw schemaInvalid();
  return value;
}

function spendMeter(value: unknown): number | undefined {
  const spent = nullableNumber(value);
  if (spent !== undefined && spent < 0) throw schemaInvalid();
  return spent;
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

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
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
