// Adaptive prefill optimizer evidence rendered from the existing status document.

function optimizerConvergenceAssessment(optimizerDocument) {
    if (!optimizerDocument || optimizerDocument.enabled === null) {
        return {
            title: "Optimizer unavailable",
            detail: "Configuration and runtime evidence have not reached the Observatory yet.",
            tone: "waiting"
        };
    }
    if (!optimizerDocument.enabled) {
        const fixedPrefillChunckTokens = optimizerDocument.fixed_prefill_chunck_tokens;
        return {
            title: "Fixed prefill size",
            detail: fixedPrefillChunckTokens
                ? formatOptimizerTokenCount(fixedPrefillChunckTokens) + " tokens is configured; adaptive learning is off."
                : "Adaptive learning is off.",
            tone: "fixed"
        };
    }
    const latestInsight = optimizerDocument.latest_insight;
    if (!latestInsight) {
        return {
            title: "Awaiting context evidence",
            detail: "The optimizer is enabled and will report evidence after prompt processing.",
            tone: "waiting"
        };
    }
    const requestedTokens = formatOptimizerTokenCount(latestInsight.requested_prefill_chunck_tokens);
    const actualTokens = formatOptimizerTokenCount(latestInsight.actual_prefill_chunck_tokens);
    if (latestInsight.has_observations_for_every_candidate) {
        return {
            title: "Context evidence ready",
            detail: requestedTokens + " tokens is preferred for the latest context, not a global fixed size.",
            tone: "ready"
        };
    }
    const memoryPressureDetail = latestInsight.has_observed_prefill_capacity_constraint
        ? requestedTokens + " requested, " + actualTokens + " completed under memory pressure."
        : requestedTokens + " tokens was requested while candidate evidence is still being collected.";
    return { title: "Still exploring", detail: memoryPressureDetail, tone: "learning" };
}

function renderPrefillOptimizer(optimizerDocument) {
    const convergenceAssessment = optimizerConvergenceAssessment(optimizerDocument);
    const assessmentPanel = document.getElementById("optimizer-assessment");
    assessmentPanel.dataset.tone = convergenceAssessment.tone;
    document.getElementById("optimizer-assessment-title").textContent = convergenceAssessment.title;
    document.getElementById("optimizer-assessment-detail").textContent = convergenceAssessment.detail;

    document.getElementById("optimizer-mode").textContent = optimizerModeTitle(optimizerDocument);
    const latestInsight = optimizerDocument ? optimizerDocument.latest_insight : null;
    document.getElementById("optimizer-requested-tokens").textContent = latestInsight
        ? formatOptimizerTokenCount(latestInsight.requested_prefill_chunck_tokens)
        : "—";
    document.getElementById("optimizer-actual-tokens").textContent = latestInsight
        ? formatOptimizerTokenCount(latestInsight.actual_prefill_chunck_tokens)
        : "—";
    document.getElementById("optimizer-elapsed").textContent = latestInsight
        ? Number(latestInsight.elapsed_millis || 0).toLocaleString() + " ms"
        : "—";
    document.getElementById("optimizer-decision-reason").textContent = latestInsight
        ? optimizerDecisionReasonTitle(latestInsight.decision_reason)
        : "No measured decision";
    renderOptimizerContext(latestInsight ? latestInsight.context : null);
    renderOptimizerCandidateEvidence(optimizerDocument, latestInsight);
    renderOptimizerTransitions(optimizerDocument ? optimizerDocument.recent_transitions : []);
}

function optimizerModeTitle(optimizerDocument) {
    if (!optimizerDocument || optimizerDocument.enabled === null) { return "Awaiting"; }
    return optimizerDocument.enabled ? "Adaptive" : "Fixed";
}

function renderOptimizerContext(optimizerContext) {
    const contextContainer = document.getElementById("optimizer-context");
    contextContainer.replaceChildren();
    if (!optimizerContext) {
        appendOptimizerPill(contextContainer, "No context observed", "muted");
        return;
    }
    appendOptimizerPill(
        contextContainer,
        "Position " + formatOptimizerTokenCount(optimizerContext.prompt_position_tokens),
        "position"
    );
    appendOptimizerPill(contextContainer, optimizerContext.has_restored_prefix ? "Restored prefix" : "Cold prefix");
    appendOptimizerPill(contextContainer, optimizerContext.has_visual_embeddings ? "Vision" : "Text only");
    appendOptimizerPill(contextContainer, optimizerContext.is_mtp_active ? "MTP active" : "Target only");
    appendOptimizerPill(contextContainer, optimizerContext.are_sparse_experts_paged ? "Experts paged" : "Experts resident");
    appendOptimizerPill(contextContainer, optimizerContext.is_prompt_cache_capture_eligible ? "Cache capture eligible" : "Cache capture unavailable");
    if (optimizerContext.is_first_chunck_after_restore) {
        appendOptimizerPill(contextContainer, "First after restore", "attention");
    }
    if (optimizerContext.has_prior_capacity_reduction) {
        appendOptimizerPill(contextContainer, "Prior memory reduction", "attention");
    }
}

function appendOptimizerPill(contextContainer, label, pillTone) {
    const contextPill = document.createElement("span");
    contextPill.className = "optimizer-context-pill" + (pillTone ? " " + pillTone : "");
    contextPill.textContent = label;
    contextContainer.appendChild(contextPill);
}

function fastestMeasuredCandidateForLatestContext(candidateEvidence) {
    let fastestMeasuredCandidate = null;
    let fastestMeasuredPrefillTokensPerSecond = 0;
    let hasEqualFastestMeasuredCandidate = false;
    candidateEvidence.forEach((candidateMeasurement) => {
        const observationCount = Number(candidateMeasurement.observation_count || 0);
        const averageActualPrefillChunckTokens = Number(
            candidateMeasurement.average_actual_prefill_chunck_tokens || 0
        );
        const averageElapsedMillis = Number(candidateMeasurement.average_elapsed_millis || 0);
        if (observationCount === 0 || averageActualPrefillChunckTokens === 0 || averageElapsedMillis === 0) {
            return;
        }
        const measuredPrefillTokensPerSecond =
            averageActualPrefillChunckTokens * 1000 / averageElapsedMillis;
        if (measuredPrefillTokensPerSecond > fastestMeasuredPrefillTokensPerSecond) {
            fastestMeasuredCandidate = candidateMeasurement;
            fastestMeasuredPrefillTokensPerSecond = measuredPrefillTokensPerSecond;
            hasEqualFastestMeasuredCandidate = false;
        } else if (measuredPrefillTokensPerSecond === fastestMeasuredPrefillTokensPerSecond) {
            hasEqualFastestMeasuredCandidate = true;
        }
    });
    return hasEqualFastestMeasuredCandidate ? null : fastestMeasuredCandidate;
}

function renderOptimizerCandidateEvidence(optimizerDocument, latestInsight) {
    const evidenceTableBody = document.getElementById("optimizer-evidence-body");
    evidenceTableBody.replaceChildren();
    const configuredCandidates = optimizerDocument
        ? optimizerDocument.candidate_prefill_chunck_tokens || []
        : [];
    const candidateEvidence = latestInsight ? latestInsight.candidate_evidence || [] : [];
    const candidatesToRender = candidateEvidence.length > 0
        ? candidateEvidence
        : configuredCandidates.map((candidatePrefillChunckTokens) => ({
            candidate_prefill_chunck_tokens: candidatePrefillChunckTokens,
            observation_count: 0,
            average_actual_prefill_chunck_tokens: 0,
            average_elapsed_millis: 0,
            decisions_since_last_observation: null
        }));
    const fastestMeasuredCandidate = fastestMeasuredCandidateForLatestContext(candidatesToRender);
    if (candidatesToRender.length === 0) {
        const emptyRow = document.createElement("tr");
        const emptyCell = document.createElement("td");
        emptyCell.colSpan = 6;
        emptyCell.textContent = "No candidate evidence available.";
        emptyRow.appendChild(emptyCell);
        evidenceTableBody.appendChild(emptyRow);
        return;
    }
    candidatesToRender.forEach((candidateMeasurement) => {
        const evidenceRow = document.createElement("tr");
        if (candidateMeasurement === fastestMeasuredCandidate) {
            const fastestMeasuredCandidateCell = document.createElement("td");
            fastestMeasuredCandidateCell.textContent =
                formatOptimizerTokenCount(candidateMeasurement.candidate_prefill_chunck_tokens)
                + " (Fastest measured)";
            fastestMeasuredCandidateCell.className = "optimizer-fastest-measured-cell";
            evidenceRow.appendChild(fastestMeasuredCandidateCell);
        } else {
            appendOptimizerTableCell(evidenceRow, formatOptimizerTokenCount(candidateMeasurement.candidate_prefill_chunck_tokens));
        }
        appendOptimizerTableCell(evidenceRow, Number(candidateMeasurement.observation_count || 0).toLocaleString());
        appendOptimizerTableCell(evidenceRow, candidateMeasurement.observation_count
            ? formatOptimizerTokenCount(candidateMeasurement.average_actual_prefill_chunck_tokens)
            : "—");
        appendOptimizerTableCell(evidenceRow, candidateMeasurement.observation_count
            ? Number(candidateMeasurement.average_elapsed_millis || 0).toLocaleString() + " ms"
            : "—");
        appendOptimizerTableCell(evidenceRow, candidateMeasurement.observation_count
            && candidateMeasurement.average_elapsed_millis
            ? Math.round(
                candidateMeasurement.average_actual_prefill_chunck_tokens * 1000
                / candidateMeasurement.average_elapsed_millis
            ).toLocaleString() + " tok/s"
            : "—");
        appendOptimizerTableCell(evidenceRow, candidateMeasurement.decisions_since_last_observation === null
            || candidateMeasurement.decisions_since_last_observation === undefined
            ? "Never"
            : Number(candidateMeasurement.decisions_since_last_observation).toLocaleString() + " decisions");
        evidenceTableBody.appendChild(evidenceRow);
    });
}

function appendOptimizerTableCell(evidenceRow, cellText) {
    const evidenceCell = document.createElement("td");
    evidenceCell.textContent = cellText;
    evidenceRow.appendChild(evidenceCell);
}

function renderOptimizerTransitions(recentTransitions) {
    const transitionList = document.getElementById("optimizer-transitions");
    transitionList.replaceChildren();
    if (!recentTransitions || recentTransitions.length === 0) {
        const emptyTransition = document.createElement("li");
        emptyTransition.textContent = "No optimizer transitions observed in this worker session.";
        transitionList.appendChild(emptyTransition);
        return;
    }
    recentTransitions.slice().reverse().forEach((optimizerTransition) => {
        const transitionListItem = document.createElement("li");
        const transitionSummary = document.createElement("strong");
        transitionSummary.textContent = formatOptimizerTokenCount(optimizerTransition.requested_prefill_chunck_tokens)
            + " requested → " + formatOptimizerTokenCount(optimizerTransition.actual_prefill_chunck_tokens) + " completed";
        const transitionDetail = document.createElement("span");
        transitionDetail.textContent = optimizerDecisionReasonTitle(optimizerTransition.decision_reason)
            + " · " + Number(optimizerTransition.elapsed_millis || 0).toLocaleString() + " ms";
        transitionListItem.append(transitionSummary, transitionDetail);
        transitionList.appendChild(transitionListItem);
    });
}

function optimizerDecisionReasonTitle(decisionReason) {
    if (decisionReason === "initial_exploration") { return "Initial exploration"; }
    if (decisionReason === "stale_observation_probe") { return "Freshness probe"; }
    if (decisionReason === "cumulative_latency_planning") { return "Cumulative latency plan"; }
    return "Fallback decision";
}

function formatOptimizerTokenCount(prefillChunckTokens) {
    return Number(prefillChunckTokens || 0).toLocaleString("en-US");
}
