import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  readCachedOpenRouterProvider,
  writeCachedProviders,
} from "../../src/cache.js";
import { openrouterCredentialContextId } from "../../src/lib/fs.js";
import {
  createOpenRouterAdapter,
  normalizeOpenRouterPayload,
  resolveOpenRouterCredential,
} from "../../src/providers/openrouter.js";
import type {
  ProviderAdapter,
  ProviderQuota,
  QuotaWindow,
} from "../../src/types.js";

const NOW = Date.parse("2026-09-16T12:00:00.000Z");
const OPTIONS = { allowKeychainPrompt: false, refreshCredentials: false };
const SYNTHETIC_KEY = "synthetic-openrouter-key-733";

const KEY_PAYLOAD = {
  data: {
    label: "sk-or-v1-synthetic",
    limit: 80,
    limit_remaining: 61.5,
    limit_reset: "daily",
    usage: 1234.56,
    usage_daily: 18.5,
    usage_weekly: 120.25,
    usage_monthly: 480.75,
    is_free_tier: false,
    is_provisioning_key: false,
    rate_limit: { requests: 40, interval: "10s" },
  },
};

describe("OpenRouter request transport", () => {
  it("makes one fixed-origin read-only request with a bearer token", async () => {
    const request = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        jsonResponse(KEY_PAYLOAD),
    );
    const adapter = testAdapter({ fetch: request });

    const report = await adapter.fetchQuota(OPTIONS);

    expect(request).toHaveBeenCalledTimes(1);
    const [input, init] = request.mock.calls[0];
    const url = new URL(String(input));
    expect({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || "443",
      pathname: url.pathname,
      search: url.search,
      hash: url.hash,
    }).toEqual({
      protocol: "https:",
      hostname: "openrouter.ai",
      port: "443",
      pathname: "/api/v1/auth/key",
      search: "",
      hash: "",
    });
    expect(init?.method).toBe("GET");
    expect(init?.redirect).toBe("manual");
    expect(init?.credentials).toBe("omit");
    const headers = new Headers(init?.headers);
    expect(headers.get("authorization")).toBe(`Bearer ${SYNTHETIC_KEY}`);
    expect(headers.get("user-agent")).toMatch(/^quota-axi\/\d+\.\d+\.\d+$/);
    expect(headers.get("cookie")).toBeNull();
    expect(report).toMatchObject({
      provider: "openrouter",
      label: "OpenRouter",
      source: "api",
      state: {
        status: "fresh",
        stale: false,
        sourcesTried: ["pi:openrouter"],
      },
      attempts: [{ source: "pi:openrouter", status: "success" }],
    });
    expect(report.account).toBeUndefined();
    expect(JSON.stringify(report)).not.toContain(SYNTHETIC_KEY);
    expect(JSON.stringify(report)).not.toContain("sk-or-v1-synthetic");
  });

  it("coalesces concurrent acquisitions into one provider request", async () => {
    let finish: ((response: Response) => void) | undefined;
    const request = vi.fn(
      async () =>
        new Promise<Response>((resolve) => {
          finish = resolve;
        }),
    );
    const adapter = testAdapter({ fetch: request });

    const first = adapter.fetchQuota(OPTIONS);
    const second = adapter.fetchQuota(OPTIONS);
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(1));
    finish?.(jsonResponse(KEY_PAYLOAD));

    const [firstReport, secondReport] = await Promise.all([first, second]);
    expect(firstReport).toBe(secondReport);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("rejects every redirect without a follow-up request", async () => {
    for (const status of [300, 301, 302, 303, 307, 308]) {
      const request = vi.fn(
        async () =>
          new Response("redirect payload", {
            status,
            headers: { location: "https://elsewhere.invalid/secret" },
          }),
      );
      const report = await testAdapter({ fetch: request }).fetchQuota(OPTIONS);
      expect(request).toHaveBeenCalledTimes(1);
      expect(report.state).toMatchObject({
        status: "error",
        stale: false,
        error: "redirect_rejected",
      });
    }
  });

  it.each([
    [401, "auth_required", "provider_auth_rejected"],
    [403, "auth_required", "provider_auth_rejected"],
    [408, "error", "provider_timeout"],
    [429, "rate_limited", "provider_rate_limited"],
    [503, "error", "provider_unavailable"],
    [418, "error", "provider_request_rejected"],
  ])(
    "maps HTTP %i to bounded status and error",
    async (status, expectedStatus, code) => {
      const report = await testAdapter({
        fetch: vi.fn(
          async () => new Response("sensitive provider text", { status }),
        ),
      }).fetchQuota(OPTIONS);

      expect(report.state.status).toBe(expectedStatus);
      expect(report.state.error).toBe(code);
      expect(JSON.stringify(report)).not.toContain("sensitive provider text");
    },
  );

  it("normalizes integer and HTTP-date Retry-After on a rate limit", async () => {
    const now = () => NOW;
    for (const [value, expected] of [
      ["91", "2026-09-16T12:01:31.000Z"],
      ["Wed, 16 Sep 2026 12:06:07 GMT", "2026-09-16T12:06:07.000Z"],
      ["soon", undefined],
    ] as const) {
      const report = await testAdapter({
        now,
        fetch: vi.fn(
          async () =>
            new Response(null, {
              status: 429,
              headers: { "retry-after": value },
            }),
        ),
      }).fetchQuota(OPTIONS);
      expect(report.state.retryAfter).toBe(expected);
    }
  });

  it("maps local failures without exposing error text", async () => {
    const sentinel = "SENTINEL-transport-secret-733159";
    const network = await testAdapter({
      fetch: vi.fn(async () => {
        throw new Error(sentinel);
      }),
    }).fetchQuota(OPTIONS);
    const tls = await testAdapter({
      fetch: vi.fn(async () => {
        throw { cause: { code: "CERT_SIGNATURE_FAILURE" }, sentinel };
      }),
    }).fetchQuota(OPTIONS);

    expect(network.state.error).toBe("network_unavailable");
    expect(tls.state.error).toBe("tls_failed");
    expect(JSON.stringify([network, tls])).not.toContain(sentinel);
  });

  it("enforces the deadline when a fetch implementation does not honor abort", async () => {
    const report = await testAdapter({
      fetch: vi.fn(async () => new Promise<Response>(() => {})),
      deadlineMs: 5,
    }).fetchQuota(OPTIONS);

    expect(report.state.error).toBe("request_timeout");
  });

  it("times out a stalled body read at the deadline and releases the reader later", async () => {
    let resolveRead:
      | ((result: ReadableStreamReadResult<Uint8Array>) => void)
      | undefined;
    let readSettled = false;
    const cancel = vi.fn(async () => undefined);
    const releaseLock = vi.fn(() => {
      if (!readSettled) throw new Error("read_pending");
    });
    const report = await testAdapter({
      deadlineMs: 10,
      fetch: vi.fn(async () =>
        stubResponse({
          getReader: () => ({
            read: () =>
              new Promise<ReadableStreamReadResult<Uint8Array>>((resolve) => {
                resolveRead = resolve;
              }),
            cancel,
            releaseLock,
          }),
        }),
      ),
    }).fetchQuota(OPTIONS);

    expect(report.state).toMatchObject({
      status: "error",
      error: "request_timeout",
    });
    expect(cancel).toHaveBeenCalledOnce();

    readSettled = true;
    resolveRead?.({ done: true, value: undefined });
    await vi.waitFor(() => expect(releaseLock).toHaveBeenCalled());
  });

  it("returns a timeout even when cancellation and the pending read both stall", async () => {
    const cancel = vi.fn(() => new Promise<void>(() => undefined));
    const startedAt = Date.now();
    const report = await testAdapter({
      deadlineMs: 10,
      fetch: vi.fn(async () =>
        stubResponse({
          getReader: () => ({
            read: () =>
              new Promise<ReadableStreamReadResult<Uint8Array>>(
                () => undefined,
              ),
            cancel,
            releaseLock: vi.fn(),
          }),
        }),
      ),
    }).fetchQuota(OPTIONS);

    expect(report.state).toMatchObject({
      status: "error",
      error: "request_timeout",
    });
    expect(cancel).toHaveBeenCalledOnce();
    expect(Date.now() - startedAt).toBeLessThan(5_000);
  });

  it("does not wait on a stalled cancellation of a rejected response body", async () => {
    const cancel = vi.fn(() => new Promise<void>(() => undefined));
    const report = await testAdapter({
      fetch: vi.fn(async () => stubResponse({ cancel }, 503)),
    }).fetchQuota(OPTIONS);

    expect(report.state.error).toBe("provider_unavailable");
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("caps decoded body bytes even when the declared length is small", async () => {
    let cancellations = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(new Uint8Array(262_145));
      },
      cancel() {
        cancellations += 1;
      },
    });
    const report = await testAdapter({
      fetch: vi.fn(
        async () => new Response(body, { headers: { "content-length": "16" } }),
      ),
    }).fetchQuota(OPTIONS);

    expect(report.state).toMatchObject({
      status: "error",
      error: "response_too_large",
    });
    await vi.waitFor(() => expect(cancellations).toBe(1));
  });

  it("does not wait on stalled reader cleanup after an oversized chunk", async () => {
    const report = await testAdapter({
      fetch: vi.fn(async () =>
        stubResponse({
          getReader: () => ({
            read: async () => ({
              done: false,
              value: new Uint8Array(262_145),
            }),
            cancel: () => new Promise<never>(() => undefined),
            releaseLock: vi.fn(),
          }),
        }),
      ),
    }).fetchQuota(OPTIONS);

    expect(report.state.error).toBe("response_too_large");
  });

  it("rejects an oversized declared length without reading and bounds its cancellation", async () => {
    const getReader = vi.fn();
    const cancel = vi.fn(() => new Promise<void>(() => undefined));
    const report = await testAdapter({
      fetch: vi.fn(async () =>
        stubResponse({ getReader, cancel }, 200, {
          "content-length": "262145",
        }),
      ),
    }).fetchQuota(OPTIONS);

    expect(report.state.error).toBe("response_too_large");
    expect(getReader).not.toHaveBeenCalled();
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("rejects invalid UTF-8, malformed JSON, and invalid schema", async () => {
    const cases: Array<[Response, string]> = [
      [
        new Response(Uint8Array.from([0xc3, 0x28]), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
        "response_invalid_utf8",
      ],
      [
        new Response("{unfinished", {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
        "malformed_json",
      ],
      [jsonResponse({ data: {} }), "schema_invalid"],
      [jsonResponse("not-an-object"), "schema_invalid"],
    ];

    for (const [response, code] of cases) {
      const report = await testAdapter({
        fetch: vi.fn(async () => response),
      }).fetchQuota(OPTIONS);
      expect(report.state.error).toBe(code);
    }
  });
});

describe("OpenRouter payload normalization", () => {
  it("maps the key record to a daily limit window plus usage meters", () => {
    expect(normalizeOpenRouterPayload(KEY_PAYLOAD, NOW)).toEqual([
      {
        id: "limit",
        label: "day",
        kind: "credits",
        percentRemaining: 76.875,
        percentUsed: 23.125,
        spentUsd: 18.5,
        limitUsd: 80,
        startsAt: "2026-09-16T00:00:00.000Z",
        resetsAt: "2026-09-17T00:00:00.000Z",
        resetText: "daily",
      },
      {
        id: "usage_daily",
        label: "day usage",
        kind: "credits",
        spentUsd: 18.5,
        startsAt: "2026-09-16T00:00:00.000Z",
        resetsAt: "2026-09-17T00:00:00.000Z",
      },
      {
        id: "usage_weekly",
        label: "week usage",
        kind: "credits",
        spentUsd: 120.25,
        startsAt: "2026-09-14T00:00:00.000Z",
        resetsAt: "2026-09-21T00:00:00.000Z",
      },
      {
        id: "usage_monthly",
        label: "month usage",
        kind: "credits",
        spentUsd: 480.75,
        startsAt: "2026-09-01T00:00:00.000Z",
        resetsAt: "2026-10-01T00:00:00.000Z",
      },
    ]);
  });

  it("resolves UTC periods at their day, Monday-week, and month boundaries", () => {
    const payload = {
      data: {
        limit: 10,
        limit_remaining: 5,
        limit_reset: "weekly",
        usage_daily: 1,
        usage_monthly: 2,
      },
    };
    const periods = (now: string) =>
      normalizeOpenRouterPayload(payload, Date.parse(now)).map(
        ({ id, startsAt, resetsAt }) => ({ id, startsAt, resetsAt }),
      );

    expect(periods("2026-12-31T23:59:59.999Z")).toEqual([
      {
        id: "limit",
        startsAt: "2026-12-28T00:00:00.000Z",
        resetsAt: "2027-01-04T00:00:00.000Z",
      },
      {
        id: "usage_daily",
        startsAt: "2026-12-31T00:00:00.000Z",
        resetsAt: "2027-01-01T00:00:00.000Z",
      },
      {
        id: "usage_monthly",
        startsAt: "2026-12-01T00:00:00.000Z",
        resetsAt: "2027-01-01T00:00:00.000Z",
      },
    ]);
    expect(periods("2026-09-20T23:00:00.000Z")[0]).toEqual({
      id: "limit",
      startsAt: "2026-09-14T00:00:00.000Z",
      resetsAt: "2026-09-21T00:00:00.000Z",
    });
    expect(periods("2026-09-21T00:00:00.000Z")[0]).toEqual({
      id: "limit",
      startsAt: "2026-09-21T00:00:00.000Z",
      resetsAt: "2026-09-28T00:00:00.000Z",
    });
  });

  it("reports only usage meters for an unlimited key", () => {
    const windows = normalizeOpenRouterPayload(
      {
        data: {
          limit: null,
          limit_remaining: null,
          limit_reset: null,
          usage: 12.5,
          usage_daily: 1.25,
          usage_weekly: 4.5,
          usage_monthly: 9.75,
        },
      },
      NOW,
    );
    expect(windows.map(({ id }) => id)).toEqual([
      "usage_daily",
      "usage_weekly",
      "usage_monthly",
    ]);
    expect(
      windows.every(({ percentRemaining }) => percentRemaining === undefined),
    ).toBe(true);
  });

  it("labels a weekly-reset limit with its current UTC week", () => {
    const [window] = normalizeOpenRouterPayload(
      {
        data: { limit: 100, limit_remaining: 25, limit_reset: "weekly" },
      },
      NOW,
    );
    expect(window).toMatchObject({
      id: "limit",
      label: "week",
      startsAt: "2026-09-14T00:00:00.000Z",
      resetsAt: "2026-09-21T00:00:00.000Z",
      resetText: "weekly",
      percentRemaining: 25,
      percentUsed: 75,
      spentUsd: 75,
      limitUsd: 100,
    });
  });

  it("resolves a monthly-reset limit to its calendar month", () => {
    const [window] = normalizeOpenRouterPayload(
      {
        data: { limit: 100, limit_remaining: 40, limit_reset: "monthly" },
      },
      NOW,
    );
    expect(window).toMatchObject({
      id: "limit",
      label: "month",
      startsAt: "2026-09-01T00:00:00.000Z",
      resetsAt: "2026-10-01T00:00:00.000Z",
      resetText: "monthly",
    });
  });

  it("treats a limit without a recognized reset cadence as a credit cap", () => {
    for (const limit_reset of [undefined, null, "fortnightly"]) {
      const [window] = normalizeOpenRouterPayload(
        {
          data: { limit: 50, limit_remaining: 20, limit_reset },
        },
        NOW,
      );
      expect(window).toMatchObject({ id: "limit", label: "credits" });
      expect(window.resetsAt).toBeUndefined();
      expect(window.resetText).toBeUndefined();
    }
  });

  it("reports a zeroed limit as fully used", () => {
    const [window] = normalizeOpenRouterPayload(
      {
        data: { limit: 0, limit_remaining: 0, limit_reset: "daily" },
      },
      NOW,
    );
    expect(window).toMatchObject({
      id: "limit",
      percentRemaining: 0,
      percentUsed: 100,
      limitUsd: 0,
    });
  });

  it("clamps an overdrawn negative remaining to zero percent", () => {
    const [window] = normalizeOpenRouterPayload(
      {
        data: { limit: 80, limit_remaining: -5, limit_reset: "daily" },
      },
      NOW,
    );
    expect(window).toMatchObject({
      id: "limit",
      percentRemaining: 0,
      percentUsed: 100,
      spentUsd: 85,
      limitUsd: 80,
    });
  });

  it.each([
    ["a record outside the data envelope", { limit: 10, limit_remaining: 5 }],
    ["a record without limit", { data: { limit_reset: "daily" } }],
    [
      "a limit without its remaining amount",
      { data: { limit: 80, limit_reset: "daily", usage_daily: 1 } },
    ],
    [
      "a non-numeric limit",
      { data: { limit: "eighty", limit_remaining: 5, usage_daily: 1 } },
    ],
    ["a negative limit", { data: { limit: -1, limit_remaining: 0 } }],
    [
      "a non-string reset cadence",
      { data: { limit: 10, limit_remaining: 5, limit_reset: 1 } },
    ],
    [
      "a non-numeric lifetime usage",
      { data: { limit: null, usage: {}, usage_daily: 1 } },
    ],
    [
      "a negative usage meter",
      { data: { limit: 10, limit_remaining: 5, usage_daily: -1 } },
    ],
    [
      "a non-numeric usage meter",
      { data: { limit: 10, limit_remaining: 5, usage_weekly: "n/a" } },
    ],
    ["an unlimited key with no usage meters", { data: { limit: null } }],
  ])("throws schema_invalid for %s", (_label, payload) => {
    expect(() => normalizeOpenRouterPayload(payload, NOW)).toThrow(
      "schema_invalid",
    );
  });

  it("throws schema_invalid for a record with no recognized field", () => {
    expect(() => normalizeOpenRouterPayload({ data: {} }, NOW)).toThrow();
    expect(() => normalizeOpenRouterPayload([], NOW)).toThrow();
    expect(() => normalizeOpenRouterPayload(undefined, NOW)).toThrow();
  });
});

describe("OpenRouter credential discovery", () => {
  it("reads a Pi api_key entry from the Pi agent auth file", () => {
    withPiAuthFile(
      JSON.stringify({
        openrouter: { type: "api_key", key: SYNTHETIC_KEY },
      }),
      (authFile) => {
        expect(resolveOpenRouterCredential(() => authFile)).toEqual({
          status: "available",
          apiKey: SYNTHETIC_KEY,
          path: authFile,
        });
      },
    );
  });

  it("reports a Pi auth file without an openrouter entry as missing", () => {
    withPiAuthFile(
      JSON.stringify({ xai: { type: "api_key", key: SYNTHETIC_KEY } }),
      (authFile) => {
        expect(resolveOpenRouterCredential(() => authFile)).toEqual({
          status: "missing",
          path: authFile,
        });
      },
    );
  });

  it.each([
    [
      "a command reference",
      { openrouter: { type: "api_key", key: "!pass show openrouter" } },
    ],
    [
      "an environment reference",
      { openrouter: { type: "api_key", key: "$OPENROUTER_API_KEY" } },
    ],
    ["a non-object entry", { openrouter: "literal-key" }],
    [
      "an unsupported type",
      { openrouter: { type: "oauth", access: SYNTHETIC_KEY } },
    ],
  ])(
    "reports a Pi entry holding %s as invalid, not missing",
    (_label, auth) => {
      withPiAuthFile(JSON.stringify(auth), (authFile) => {
        expect(resolveOpenRouterCredential(() => authFile)).toEqual({
          status: "invalid",
          path: authFile,
          error: "invalid_credential",
        });
      });
    },
  );

  it("resolves a missing auth file to missing", () => {
    const directory = mkdtempSync(join(tmpdir(), "quota-axi-openrouter-"));
    try {
      const authFile = join(directory, "auth.json");
      expect(resolveOpenRouterCredential(() => authFile)).toEqual({
        status: "missing",
        path: authFile,
      });
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("resolves an unreadable auth file to an error, not an invalid credential", () => {
    const directory = mkdtempSync(join(tmpdir(), "quota-axi-openrouter-"));
    try {
      const authFile = join(directory, "auth.json");
      mkdirSync(authFile);
      expect(resolveOpenRouterCredential(() => authFile)).toEqual({
        status: "error",
        path: authFile,
        error: "file_read_error",
      });
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("resolves a malformed auth file to an invalid credential", () => {
    withPiAuthFile("{broken", (authFile) => {
      expect(resolveOpenRouterCredential(() => authFile)).toEqual({
        status: "invalid",
        path: authFile,
        error: "json_parse_error",
      });
    });
  });

  it("makes no request and retires cache for missing credentials", async () => {
    const request = vi.fn();
    const remove = vi.fn();
    const report = await testAdapter({
      credential: () => ({ status: "missing", path: PI_PATH }),
      fetch: request,
      deleteCachedProvider: remove,
      readCachedProvider: () => cachedQuota(),
    }).fetchQuota(OPTIONS);

    expect(request).not.toHaveBeenCalled();
    expect(remove).toHaveBeenCalledWith(openrouterCredentialContextId(PI_PATH));
    expect(report.state).toMatchObject({
      status: "auth_required",
      stale: false,
      error: "openrouter_credential_unavailable",
    });
    expect(report.windows).toEqual([]);
    expect(report.attempts).toEqual([
      {
        source: "pi:openrouter",
        status: "skipped",
        error: "openrouter_credential_unavailable",
      },
    ]);
  });

  it("makes no request and retires cache for invalid credentials", async () => {
    const request = vi.fn();
    const remove = vi.fn();
    const report = await testAdapter({
      credential: () => ({
        status: "invalid",
        path: PI_PATH,
        error: "invalid_credential",
      }),
      fetch: request,
      deleteCachedProvider: remove,
      readCachedProvider: () => cachedQuota(),
    }).fetchQuota(OPTIONS);

    expect(request).not.toHaveBeenCalled();
    expect(remove).toHaveBeenCalledWith(openrouterCredentialContextId(PI_PATH));
    expect(report.state).toMatchObject({
      status: "auth_required",
      error: "openrouter_credential_invalid",
    });
  });

  it("serves stale cache and keeps it when the auth file cannot be read", async () => {
    const request = vi.fn();
    const remove = vi.fn();
    const report = await testAdapter({
      credential: () => ({
        status: "error",
        path: PI_PATH,
        error: "file_read_error",
      }),
      fetch: request,
      deleteCachedProvider: remove,
      readCachedProvider: () => cachedQuota(),
    }).fetchQuota(OPTIONS);

    expect(request).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();
    expect(report.source).toBe("cache");
    expect(report.state).toMatchObject({
      status: "stale",
      stale: true,
      error: "credential_resolution_failed",
      sourcesTried: ["pi:openrouter", "cache"],
    });
    expect(report.windows.length).toBeGreaterThan(0);
  });
});

describe("OpenRouter cache fallback", () => {
  it("drops cache on a definitive 401 auth rejection", async () => {
    const remove = vi.fn();
    const report = await testAdapter({
      fetch: vi.fn(async () => new Response(null, { status: 401 })),
      deleteCachedProvider: remove,
      readCachedProvider: () => cachedQuota(),
    }).fetchQuota(OPTIONS);

    expect(remove).toHaveBeenCalledWith(openrouterCredentialContextId(PI_PATH));
    expect(report.state).toMatchObject({
      status: "auth_required",
      error: "provider_auth_rejected",
    });
    expect(report.source).toBe("unavailable");
    expect(report.windows).toEqual([]);
  });

  it("uses stale cache for transient HTTP failures", async () => {
    for (const status of [408, 502, 503]) {
      const report = await testAdapter({
        fetch: vi.fn(async () => new Response(null, { status })),
        readCachedProvider: () => cachedQuota(),
      }).fetchQuota(OPTIONS);
      expect(report.state.status).toBe("stale");
      expect(report.source).toBe("cache");
    }
  });

  it("does not fall back to cache for malformed responses", async () => {
    const report = await testAdapter({
      fetch: vi.fn(
        async () =>
          new Response("{broken", {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
      ),
      readCachedProvider: () => cachedQuota(),
    }).fetchQuota(OPTIONS);

    expect(report.state.status).toBe("error");
    expect(report.state.error).toBe("malformed_json");
    expect(report.source).toBe("unavailable");
  });

  it("keeps the previous snapshot when a recognized field is malformed", async () => {
    await withPiProfiles(async ({ profileA }) => {
      process.env.PI_CODING_AGENT_DIR = profileA;
      const contextId = openrouterCredentialContextId();
      writeCachedProviders([await liveAdapter(jsonResponse(KEY_PAYLOAD))]);
      expect(readCachedOpenRouterProvider(contextId)).toBeDefined();

      for (const payload of [
        { data: { limit_reset: "daily" } },
        { data: { ...KEY_PAYLOAD.data, usage_daily: "unknown" } },
      ]) {
        const report = await liveAdapter(jsonResponse(payload));
        expect(report.state).toMatchObject({
          status: "error",
          error: "schema_invalid",
        });
        writeCachedProviders([report]);
        expect(readCachedOpenRouterProvider(contextId)).toBeDefined();
      }
    });
  });

  it("never reads or retires another Pi profile's snapshot", async () => {
    await withPiProfiles(async ({ profileA, profileB }) => {
      process.env.PI_CODING_AGENT_DIR = profileA;
      const contextA = openrouterCredentialContextId();
      writeCachedProviders([await liveAdapter(jsonResponse(KEY_PAYLOAD))]);

      process.env.PI_CODING_AGENT_DIR = profileB;
      for (const response of [
        () => new Response(null, { status: 429 }),
        () => new Response(null, { status: 503 }),
        () => new Response(null, { status: 401 }),
      ]) {
        const report = await liveAdapter(response());
        expect(report.source).toBe("unavailable");
        expect(report.windows).toEqual([]);
        expect(readCachedOpenRouterProvider(contextA)).toBeDefined();
      }
      const timedOut = await liveAdapter(
        new Promise<Response>(() => undefined),
      );
      expect(timedOut.state.error).toBe("request_timeout");
      expect(timedOut.windows).toEqual([]);

      process.env.PI_CODING_AGENT_DIR = profileA;
      const stale = await liveAdapter(new Response(null, { status: 503 }));
      expect(stale.source).toBe("cache");
      expect(stale.state.status).toBe("stale");

      await liveAdapter(new Response(null, { status: 401 }));
      expect(readCachedOpenRouterProvider(contextA)).toBeUndefined();
    });
  });

  it("expires cached windows at their UTC reset boundaries", async () => {
    const lateReading = Date.parse("2026-09-15T23:30:00.000Z");
    const windows = normalizeOpenRouterPayload(
      { data: { ...KEY_PAYLOAD.data, limit_remaining: 0 } },
      lateReading,
    );

    const beforeMidnight = await testAdapter({
      fetch: vi.fn(async () => new Response(null, { status: 503 })),
      readCachedProvider: () => cachedQuota(windows, lateReading),
      now: () => Date.parse("2026-09-15T23:59:00.000Z"),
    }).fetchQuota(OPTIONS);
    expect(beforeMidnight.windows.map(({ id }) => id)).toEqual([
      "limit",
      "usage_daily",
      "usage_weekly",
      "usage_monthly",
    ]);

    const afterMidnight = await testAdapter({
      fetch: vi.fn(async () => new Response(null, { status: 503 })),
      readCachedProvider: () => cachedQuota(windows, lateReading),
      now: () => Date.parse("2026-09-16T00:15:00.000Z"),
    }).fetchQuota(OPTIONS);
    expect(afterMidnight.state.status).toBe("stale");
    expect(afterMidnight.windows.map(({ id }) => id)).toEqual([
      "usage_weekly",
      "usage_monthly",
    ]);

    const afterMonth = await testAdapter({
      fetch: vi.fn(async () => new Response(null, { status: 503 })),
      readCachedProvider: () => cachedQuota(windows, lateReading),
      now: () => Date.parse("2026-10-01T00:00:00.000Z"),
    }).fetchQuota(OPTIONS);
    expect(afterMonth.state.status).toBe("error");
    expect(afterMonth.windows).toEqual([]);
  });

  it("bounds a cadence-less cached limit window at a month", async () => {
    const capWindow: QuotaWindow = {
      id: "limit",
      label: "credits",
      kind: "credits",
      percentUsed: 40,
      percentRemaining: 60,
      limitUsd: 100,
    };
    const fresh = await transientWithCache(
      cachedQuota([capWindow], NOW - 86_400_000),
    );
    expect(fresh.windows.map(({ id }) => id)).toEqual(["limit"]);

    const expired = await transientWithCache(
      cachedQuota([capWindow], NOW - 30 * 24 * 60 * 60 * 1_000),
    );
    expect(expired.windows).toEqual([]);
  });
});

describe("OpenRouter auth inspection", () => {
  it.each([
    ["available", "available", undefined],
    ["missing", "missing", undefined],
    ["invalid", "invalid", "invalid_credential"],
    ["error", "error", "file_read_error"],
  ] as const)(
    "reports %s credential state with the probed path and no value",
    async (status, expectedStatus, error) => {
      const report = await testAdapter({
        credential: () =>
          status === "available"
            ? { status: "available", apiKey: SYNTHETIC_KEY, path: PI_PATH }
            : status === "missing"
              ? { status: "missing", path: PI_PATH }
              : { status, path: PI_PATH, error: error ?? "invalid_credential" },
      }).inspectAuth(OPTIONS);

      expect(report).toEqual({
        provider: "openrouter",
        sources: [
          {
            source: "pi:openrouter",
            path: PI_PATH,
            status: expectedStatus,
            ...(error ? { error } : {}),
          },
        ],
      });
      expect(JSON.stringify(report)).not.toContain(SYNTHETIC_KEY);
    },
  );
});

const PI_PATH = "/home/user/.pi/agent/auth.json";
const originalXdgCacheHome = process.env.XDG_CACHE_HOME;
const originalPiAgentDir = process.env.PI_CODING_AGENT_DIR;

afterEach(() => {
  restoreEnv("XDG_CACHE_HOME", originalXdgCacheHome);
  restoreEnv("PI_CODING_AGENT_DIR", originalPiAgentDir);
});

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

/** Two Pi profiles, each holding its own key, sharing one temporary cache. */
async function withPiProfiles(
  assertion: (profiles: {
    profileA: string;
    profileB: string;
  }) => Promise<void>,
): Promise<void> {
  const directory = mkdtempSync(join(tmpdir(), "quota-axi-openrouter-"));
  try {
    process.env.XDG_CACHE_HOME = join(directory, "cache");
    const profiles = {
      profileA: join(directory, "profile-a"),
      profileB: join(directory, "profile-b"),
    };
    for (const profile of Object.values(profiles)) {
      mkdirSync(profile);
      writeFileSync(
        join(profile, "auth.json"),
        JSON.stringify({ openrouter: { type: "api_key", key: SYNTHETIC_KEY } }),
      );
    }
    await assertion(profiles);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

/** The adapter with its real credential and cache boundaries and one stubbed response. */
function liveAdapter(
  response: Response | Promise<Response>,
): Promise<ProviderQuota> {
  return createOpenRouterAdapter({
    fetch: vi.fn(async () => response) as unknown as typeof fetch,
    now: () => NOW,
    deadlineMs: 10,
  }).fetchQuota(OPTIONS);
}

function stubResponse(
  body: Record<string, unknown>,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: new Headers(headers),
    body,
  } as unknown as Response;
}

function testAdapter(
  overrides: Partial<Parameters<typeof createOpenRouterAdapter>[0]> = {},
): ProviderAdapter {
  return createOpenRouterAdapter({
    credential: () => ({
      status: "available",
      apiKey: SYNTHETIC_KEY,
      path: PI_PATH,
    }),
    fetch: vi.fn(async () =>
      jsonResponse(KEY_PAYLOAD),
    ) as unknown as typeof fetch,
    readCachedProvider: () => undefined,
    deleteCachedProvider: () => undefined,
    now: () => NOW,
    ...overrides,
  });
}

function withPiAuthFile(
  contents: string,
  assertion: (authFile: string) => void,
): void {
  const directory = mkdtempSync(join(tmpdir(), "quota-axi-openrouter-"));
  try {
    const authFile = join(directory, "auth.json");
    writeFileSync(authFile, contents);
    assertion(authFile);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function limitWindow(): QuotaWindow {
  return {
    id: "limit",
    label: "day",
    kind: "credits",
    percentUsed: 23.125,
    percentRemaining: 76.875,
    spentUsd: 18.5,
    limitUsd: 80,
    startsAt: "2026-09-16T00:00:00.000Z",
    resetsAt: "2026-09-17T00:00:00.000Z",
    resetText: "daily",
  };
}

function usageWindow(
  id: "usage_daily" | "usage_weekly" | "usage_monthly",
): QuotaWindow {
  const labels = {
    usage_daily: "day usage",
    usage_weekly: "week usage",
    usage_monthly: "month usage",
  } as const;
  return {
    id,
    label: labels[id],
    kind: "credits",
    spentUsd: 10,
    resetsAt: "2026-10-01T00:00:00.000Z",
  };
}

function cachedQuota(
  windows: QuotaWindow[] = [limitWindow(), usageWindow("usage_monthly")],
  refreshedAt = NOW - 60_000,
): ProviderQuota {
  return {
    provider: "openrouter",
    label: "OpenRouter",
    source: "api",
    windows,
    state: {
      status: "fresh",
      stale: false,
      refreshedAt: new Date(refreshedAt).toISOString(),
      sourcesTried: ["pi:openrouter"],
    },
  };
}

async function transientWithCache(
  cached: ProviderQuota,
): Promise<ProviderQuota> {
  return testAdapter({
    fetch: vi.fn(async () => new Response(null, { status: 503 })),
    readCachedProvider: () => cached,
  }).fetchQuota(OPTIONS);
}
