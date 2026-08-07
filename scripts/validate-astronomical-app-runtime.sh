# shellcheck shell=sh
# shellcheck disable=SC2154
# Runtime checks sourced by validate-astronomical-app.sh after it defines shared
# configuration, cleanup state, and progress-reporting helpers.

# Refuses to replace a reachable daemon until its active request finishes.
wait_for_running_daemon_idle_before_replacement() {
    running_daemon_pids="$(pgrep -x "astronomicald" 2>/dev/null || true)"
    if [ -z "${running_daemon_pids:-}" ]; then
        printf '%s step=wait-for-running-daemon-idle status=skipped reason=no-running-daemon\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')"
        return 0
    fi

    running_daemon_health="$(curl --silent --connect-timeout 1 --max-time 2 "${SUPERVISOR_BASE_URL}/health" 2>/dev/null || true)"
    if [ "$running_daemon_health" != "ok" ]; then
        printf '%s step=wait-for-running-daemon-idle status=skipped reason=daemon-unreachable\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')"
        return 0
    fi

    printf '%s step=wait-for-running-daemon-idle status=start timeout_seconds=%s poll_interval=%ss\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$RUNNING_DAEMON_IDLE_TIMEOUT_SECONDS" "$POLL_INTERVAL_SECONDS"
    waited_seconds=0
    while [ "$waited_seconds" -lt "$RUNNING_DAEMON_IDLE_TIMEOUT_SECONDS" ]; do
        running_status_response="$(curl --silent --connect-timeout 2 --max-time 5 "${SUPERVISOR_BASE_URL}/v1/status" 2>/dev/null || true)"
        running_activity="$(printf '%s' "$running_status_response" | jq -r '.activity // empty' 2>/dev/null || true)"
        if [ "$running_activity" = "idle" ]; then
            printf '%s step=wait-for-running-daemon-idle status=success elapsed_seconds=%s\n' \
                "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$waited_seconds"
            return 0
        fi
        printf '  [%ss/%ss] waiting for active request activity=%s\n' \
            "$waited_seconds" "$RUNNING_DAEMON_IDLE_TIMEOUT_SECONDS" "${running_activity:-unknown}"
        sleep "$POLL_INTERVAL_SECONDS"
        waited_seconds=$((waited_seconds + POLL_INTERVAL_SECONDS))
    done

    print_error "refusing to replace a daemon that remained active for ${RUNNING_DAEMON_IDLE_TIMEOUT_SECONDS}s"
    return 1
}

launch_bundled_daemon() {
    start_step "launch-daemon"
    "$daemon_executable" </dev/null >/dev/null 2>&1 &
    LAUNCHED_DAEMON_PID=$!
    printf '  launched daemon PID=%s with worker=%s\n' "$LAUNCHED_DAEMON_PID" "$worker_executable"
    sleep 2

    if ! kill -0 "$LAUNCHED_DAEMON_PID" 2>/dev/null; then
        quick_status="$(curl --silent --connect-timeout 1 --max-time 2 "${SUPERVISOR_BASE_URL}/health" 2>/dev/null || true)"
        if [ "$quick_status" != "ok" ]; then
            print_error "daemon process exited immediately; check logs in ~/.astronomical/logs/"
            finish_step "launch-daemon" "failed"
            return 1
        fi
    fi

    finish_step "launch-daemon" "success"
    return 0
}

# Polls /v1/status while reporting state transitions and bounded diagnostics.
wait_for_daemon_ready() {
    printf '%s step=wait-for-daemon-ready status=start timeout_seconds=%s poll_interval=%ss stuck_detection=%ss\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$DAEMON_STARTUP_TIMEOUT_SECONDS" "$POLL_INTERVAL_SECONDS" "$STUCK_DETECTION_SECONDS"

    waited_seconds=0
    last_status_value=""
    last_status_change_seconds=0

    while [ "$waited_seconds" -lt "$DAEMON_STARTUP_TIMEOUT_SECONDS" ]; do
        if [ -n "${LAUNCHED_DAEMON_PID:-}" ] && ! kill -0 "$LAUNCHED_DAEMON_PID" 2>/dev/null; then
            quick_check="$(curl --silent --connect-timeout 1 --max-time 2 "${SUPERVISOR_BASE_URL}/health" 2>/dev/null || true)"
            if [ "$quick_check" != "ok" ]; then
                print_error "daemon process (PID ${LAUNCHED_DAEMON_PID}) exited during startup"
                printf '  check logs: ~/.astronomical/logs/\n'
                latest_supervisor_log="$(ls -t ~/.astronomical/logs/supervisor.*.log 2>/dev/null | head -1)"
                if [ -n "$latest_supervisor_log" ]; then
                    printf '  last 10 supervisor log lines:\n'
                    tail -10 "$latest_supervisor_log" 2>/dev/null | sed 's/^/    /' || true
                fi
                latest_worker_log="$(ls -t ~/.astronomical/logs/worker.*.log 2>/dev/null | head -1)"
                if [ -n "$latest_worker_log" ]; then
                    printf '  last 10 worker log lines:\n'
                    tail -10 "$latest_worker_log" 2>/dev/null | sed 's/^/    /' || true
                fi
                return 1
            fi
        fi

        status_response="$(curl --silent --connect-timeout 2 --max-time 5 "${SUPERVISOR_BASE_URL}/v1/status" 2>/dev/null || true)"
        if [ -n "$status_response" ]; then
            status_value="$(printf '%s' "$status_response" | jq -r '.status // empty' 2>/dev/null || true)"
            activity_value="$(printf '%s' "$status_response" | jq -r '.activity // empty' 2>/dev/null || true)"

            if [ "$status_value" = "ready" ] && [ "$activity_value" = "idle" ]; then
                printf '%s step=wait-for-daemon-ready status=success elapsed_seconds=%s\n' \
                    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$waited_seconds"
                return 0
            fi

            seconds_since_status_change=$((waited_seconds - last_status_change_seconds))
            if [ "$status_value" != "$last_status_value" ]; then
                last_status_value="$status_value"
                last_status_change_seconds="$waited_seconds"
                printf '  [%ss/%ss] status=%s activity=%s\n' \
                    "$waited_seconds" "$DAEMON_STARTUP_TIMEOUT_SECONDS" \
                    "${status_value:-unknown}" "${activity_value:-unknown}"
            elif [ "$seconds_since_status_change" -ge "$STUCK_DETECTION_SECONDS" ]; then
                printf '  [%ss/%ss] STUCK on status=%s for %ss — investigating...\n' \
                    "$waited_seconds" "$DAEMON_STARTUP_TIMEOUT_SECONDS" \
                    "${status_value:-unknown}" "$seconds_since_status_change"
                worker_pids="$(pgrep -x "astronomical-inference-worker" 2>/dev/null || true)"
                if [ -z "${worker_pids:-}" ]; then
                    print_error "worker process is NOT running — daemon is stuck without a worker"
                    printf '  this usually means the worker crashed during model loading\n'
                    latest_worker_log="$(ls -t ~/.astronomical/logs/worker.*.log 2>/dev/null | head -1)"
                    if [ -n "$latest_worker_log" ]; then
                        printf '  last 15 worker log lines:\n'
                        tail -15 "$latest_worker_log" 2>/dev/null | sed 's/^/    /' || true
                    fi
                    return 1
                fi

                latest_worker_log="$(ls -t ~/.astronomical/logs/worker.*.log 2>/dev/null | head -1)"
                if [ -n "$latest_worker_log" ]; then
                    recent_errors="$(tail -30 "$latest_worker_log" 2>/dev/null | grep -i -c 'ERROR\|panic\|fatal\|crash' || true)"
                    if [ "${recent_errors:-0}" -gt 0 ]; then
                        printf '  WARNING: found %s error lines in recent worker log\n' "$recent_errors"
                        tail -15 "$latest_worker_log" 2>/dev/null | grep -i 'ERROR\|panic\|fatal\|crash' | sed 's/^/    /' || true
                    fi
                fi

                worker_cpu="$(ps -o %cpu= -p "$worker_pids" 2>/dev/null | tr -d ' ' || true)"
                printf '  worker PID=%s cpu=%s%%\n' "$worker_pids" "${worker_cpu:-?}"
                last_status_change_seconds="$waited_seconds"
            fi
        fi

        sleep "$POLL_INTERVAL_SECONDS"
        waited_seconds=$((waited_seconds + POLL_INTERVAL_SECONDS))
    done

    print_error "daemon did not become ready within ${DAEMON_STARTUP_TIMEOUT_SECONDS}s"
    return 1
}

# Verifies one configured model can serve a chat request.
validate_chat_completion() {
    start_step "validate-chat-completion"

    validation_prompt_content='Say the word "hello" and nothing else.'
    request_body_file="${VALIDATION_TEMP_DIR}/chat-request.json"
    jq -n \
        --arg model_id "$VALIDATION_MODEL_ID" \
        --arg prompt_content "$validation_prompt_content" \
        --argjson maximum_output_tokens "$CHAT_MAX_TOKENS" \
        '{
            model: $model_id,
            messages: [{role: "user", content: $prompt_content}],
            max_tokens: $maximum_output_tokens,
            temperature: 0.1,
            stream: false
        }' > "$request_body_file"

    printf '  testing configured model id=%s (max_tokens=%s, timeout=%ss)...\n' \
        "$VALIDATION_MODEL_ID" "$CHAT_MAX_TOKENS" "$CHAT_COMPLETION_TIMEOUT_SECONDS"
    chat_response="$(curl --silent --max-time "$CHAT_COMPLETION_TIMEOUT_SECONDS" \
        --header "Content-Type: application/json" \
        --data @"$request_body_file" \
        "${SUPERVISOR_BASE_URL}/v1/chat/completions" 2>&1)" || true

    chat_error="$(printf '%s' "$chat_response" | jq -r '.error.message // empty' 2>/dev/null || true)"
    if [ -z "${chat_response:-}" ] || [ -n "$chat_error" ]; then
        print_error "validation model failed: ${chat_error:-no response}"
        finish_step "validate-chat-completion" "failed"
        return 1
    fi

    response_text="$(printf '%s' "$chat_response" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)"
    if [ -z "$response_text" ]; then
        response_text="$(printf '%s' "$chat_response" | jq -r '.choices[0].message.reasoning_content // empty' 2>/dev/null || true)"
    fi
    if [ -z "$response_text" ]; then
        print_error "validation model returned empty content"
        finish_step "validate-chat-completion" "failed"
        return 1
    fi

    prompt_tokens="$(printf '%s' "$chat_response" | jq -r '.usage.prompt_tokens // empty' 2>/dev/null || true)"
    completion_tokens="$(printf '%s' "$chat_response" | jq -r '.usage.completion_tokens // empty' 2>/dev/null || true)"
    display_text="$(printf '%s' "$response_text" | cut -c1-200)"
    printf '  model replied: "%s"\n' "$display_text"
    printf '  prompt_tokens=%s completion_tokens=%s\n' "${prompt_tokens:-?}" "${completion_tokens:-?}"

    finish_step "validate-chat-completion" "success"
    return 0
}
