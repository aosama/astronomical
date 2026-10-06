import Foundation;

/// Shared helpers for fake worker processes used by the supervisor tests.
///
/// The fake workers are short `/bin/bash` scripts that emit genuine
/// length-delimited JSON frames over a real pipe pair, so the supervisor's
/// framing, decoding, and event paths are exercised end to end without any
/// model load.
enum FakeWorkerEventEmitter {

    /// A bash function that writes one frame: a four-byte big-endian length
    /// prefix followed by the payload bytes. `printf -v` computes the prefix
    /// in the shell process itself; a `$(printf ...)` command substitution
    /// would fork per frame, and those forks fail under the parallel test
    /// runner's process-table pressure, killing the fake worker before it
    /// emits anything.
    static func frameEmitterFunction() -> String {
        return "emit_frame() {\n"
            + "  payload=\"$1\"\n"
            + "  length=\"${#payload}\"\n"
            + "  printf -v prefix '\\\\x%02x\\\\x%02x\\\\x%02x\\\\x%02x' "
            + "$((length>>24&255)) $((length>>16&255)) $((length>>8&255)) $((length&255))\n"
            + "  printf \"$prefix%s\" \"$payload\"\n"
            + "}\n";
    }

    /// Wraps one event payload into a complete emit line for a script.
    static func emitLine(payload: String) -> String {
        return "emit_frame '\(payload)'\n";
    }

    /// The idle lifecycle event of a model-less worker.
    static func idleEventPayload() -> String {
        return "{\"kind\":\"idle\",\"machine_mlx_memory_ceiling_bytes\":17179869184,"
            + "\"effective_mlx_memory_ceiling_bytes\":8589934592,\"minimum_mlx_memory_ceiling_bytes\":1}";
    }

    /// A startup runtime-policy acknowledgement for a model-less worker.
    static func modelLessRuntimePolicyPayload() -> String {
        return "{\"kind\":\"runtime_feature_configuration_applied\","
            + "\"worker_runtime_feature_configuration\":{\"configuration_generation\":\"gen-1\","
            + "\"persistent_prompt_cache_enabled\":true,\"prompt_cache_maximum_size_bytes\":1073741824,"
            + "\"loaded_model\":null}}";
    }
}
