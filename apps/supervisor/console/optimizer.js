// Explains context-dependent prompt-processing chunk selection in user terms.

function optimizerAssessment(optimizerDocument) {
    const optimizerMode = optimizerDocument ? optimizerDocument.mode : "unavailable";
    if (optimizerMode === "fixed") {
        const fixedChunkSizeTokens = optimizerDocument.fixed_chunk_size_token_count;
        return {
            title: "One fixed chunk size is configured",
            detail: fixedChunkSizeTokens
                ? formatOptimizerTokenCount(fixedChunkSizeTokens)
                    + " tokens is used throughout the context window; adaptive learning is off."
                : "Adaptive learning is off.",
            tone: "fixed"
        };
    }
    if (optimizerMode !== "adaptive") {
        return {
            title: "Optimizer information is unavailable",
            detail: "Astronomical cannot show chunk-size learning until optimizer configuration is available.",
            tone: "waiting"
        };
    }
    const latestChunkOutcome = optimizerDocument.latest_chunk_outcome;
    if (!latestChunkOutcome) {
        return {
            title: "Learning starts with the next prompt",
            detail: "Astronomical will compare chunk sizes for each material execution profile.",
            tone: "waiting"
        };
    }
    const rangeTitle = optimizerPositionRangeTitle(latestChunkOutcome.measurement_context);
    if (latestChunkOutcome.was_reduced_by_memory_capacity) {
        return {
            title: "Memory capacity changed the latest decision",
            detail: "The requested chunk did not fit, so Astronomical processed a smaller amount without treating it as ordinary timing evidence.",
            tone: "learning"
        };
    }
    if (latestChunkOutcome.is_execution_profile_converged) {
        return {
            title: "Execution profile converged",
            detail: "Every configured chunk size has usable evidence. Position ranges remain observation telemetry and do not restart learning.",
            tone: "ready"
        };
    }
    const contextScope = optimizerContextScope(latestChunkOutcome, null);
    const measuredCandidateCount = contextScope
        ? contextScope.measuredProfileCandidateCount
        : 0;
    const candidateCount = contextScope && contextScope.candidateCount > 0
        ? contextScope.candidateCount
        : (optimizerDocument.candidate_chunk_size_token_counts || []).length;
    return {
        title: "Learning " + rangeTitle,
        detail: measuredCandidateCount + " of " + candidateCount
            + " chunk sizes have usable evidence for this context profile.",
        tone: "learning"
    };
}

function renderPromptProcessingOptimizer(optimizerDocument, maximumInputTokens) {
    const assessment = optimizerAssessment(optimizerDocument);
    const assessmentPanel = document.getElementById("optimizer-assessment");
    assessmentPanel.dataset.tone = assessment.tone;
    document.getElementById("optimizer-assessment-title").textContent = assessment.title;
    document.getElementById("optimizer-assessment-detail").textContent = assessment.detail;
    document.getElementById("optimizer-mode").textContent = optimizerModeTitle(optimizerDocument);

    const latestChunkOutcome = optimizerDocument ? optimizerDocument.latest_chunk_outcome : null;
    const contextScope = optimizerContextScope(latestChunkOutcome, maximumInputTokens);
    renderOptimizerCoverage(contextScope);
    renderOptimizerContextWindow(contextScope);
    renderOptimizerContext(latestChunkOutcome ? latestChunkOutcome.measurement_context : null);
    renderOptimizerCandidateMeasurements(optimizerDocument, latestChunkOutcome);
    renderOptimizerLatestDecision(latestChunkOutcome);
    renderOptimizerChunkOutcomes(optimizerDocument ? optimizerDocument.recent_chunk_outcomes : []);
}

function optimizerModeTitle(optimizerDocument) {
    if (!optimizerDocument || optimizerDocument.mode === "unavailable") { return "Unavailable"; }
    if (optimizerDocument.mode === "fixed") { return "Fixed sizing"; }
    if (optimizerDocument.mode === "adaptive") { return "Adaptive by context"; }
    return "Unavailable";
}

// Converts absolute token positions into one bounded context-window track and
// counts how much evidence belongs to the material execution profile.
function optimizerContextScope(latestChunkOutcome, maximumInputTokens) {
    const measurementContext = latestChunkOutcome ? latestChunkOutcome.measurement_context : null;
    if (!measurementContext) { return null; }
    const rangeStartTokenPosition = Math.max(
        0,
        Number(measurementContext.position_range_start_token_position || 0)
    );
    const rangeEndTokenPositionExclusive = Math.max(
        rangeStartTokenPosition,
        Number(measurementContext.position_range_end_token_position_exclusive || 0)
    );
    const chunkStartTokenPosition = Math.max(
        0,
        Number(measurementContext.chunk_start_token_position || 0)
    );
    const contextWindowTokenCount = Math.max(
        Number(maximumInputTokens || 0),
        rangeEndTokenPositionExclusive,
        chunkStartTokenPosition
    );
    const candidateMeasurements = latestChunkOutcome.candidate_measurement_summaries || [];
    const measuredProfileCandidateCount = candidateMeasurements.filter(
        (candidateMeasurement) => candidateMeasurement.measurement_count > 0
            && candidateMeasurement.measurement_source === "execution_profile"
    ).length;
    const unmeasuredCandidateCount = candidateMeasurements.length
        - measuredProfileCandidateCount;
    return {
        rangeStartTokenPosition,
        rangeEndTokenPositionExclusive,
        chunkStartTokenPosition,
        contextWindowTokenCount,
        rangeStartPercentage: optimizerTrackPercentage(
            rangeStartTokenPosition,
            contextWindowTokenCount
        ),
        rangeWidthPercentage: optimizerTrackPercentage(
            rangeEndTokenPositionExclusive - rangeStartTokenPosition,
            contextWindowTokenCount
        ),
        chunkPositionPercentage: optimizerTrackPercentage(
            chunkStartTokenPosition,
            contextWindowTokenCount
        ),
        candidateCount: candidateMeasurements.length,
        measuredProfileCandidateCount,
        unmeasuredCandidateCount
    };
}

function optimizerTrackPercentage(tokenPosition, contextWindowTokenCount) {
    if (contextWindowTokenCount <= 0) { return 0; }
    return Math.max(0, Math.min(100, tokenPosition * 100 / contextWindowTokenCount));
}

// Coverage reports usable profile evidence rather than raw position observations.

function renderOptimizerCoverage(contextScope) {
    const coverageValue = document.getElementById("optimizer-coverage-value");
    const coverageDetail = document.getElementById("optimizer-coverage-detail");
    if (!contextScope || contextScope.candidateCount === 0) {
        coverageValue.textContent = "No evidence yet";
        coverageDetail.textContent = "Candidate comparisons appear after prompt processing.";
        return;
    }
    const usableEvidenceCount = contextScope.measuredProfileCandidateCount;
    coverageValue.textContent = usableEvidenceCount + " of " + contextScope.candidateCount;
    coverageDetail.textContent = contextScope.measuredProfileCandidateCount + " measured for this profile · "
        + contextScope.unmeasuredCandidateCount + " not measured";
}

function renderOptimizerContextWindow(contextScope) {
    const contextPanel = document.getElementById("optimizer-context-window");
    const rangeBand = document.getElementById("optimizer-context-range-band");
    const chunkMarker = document.getElementById("optimizer-context-chunk-marker");
    if (!contextScope) {
        contextPanel.dataset.state = "empty";
        document.getElementById("optimizer-context-range-title").textContent = "No context observed yet";
        document.getElementById("optimizer-context-range-detail").textContent =
            "Run a prompt to see where its chunk-size evidence applies.";
        rangeBand.style.left = "0%";
        rangeBand.style.width = "0%";
        chunkMarker.style.left = "0%";
        chunkMarker.hidden = true;
        document.getElementById("optimizer-context-window-start").textContent = "0";
        document.getElementById("optimizer-context-window-end").textContent = "Context limit unavailable";
        return;
    }
    contextPanel.dataset.state = "ready";
    const rangeEndInclusive = Math.max(
        contextScope.rangeStartTokenPosition,
        contextScope.rangeEndTokenPositionExclusive - 1
    );
    document.getElementById("optimizer-context-range-title").textContent =
        "Evidence for tokens " + formatOptimizerTokenCount(contextScope.rangeStartTokenPosition)
        + "–" + formatOptimizerTokenCount(rangeEndInclusive);
    document.getElementById("optimizer-context-range-detail").textContent =
        "Chunk sizes are learned independently as prompts move into later token ranges.";
    rangeBand.style.left = contextScope.rangeStartPercentage + "%";
    rangeBand.style.width = contextScope.rangeWidthPercentage + "%";
    chunkMarker.style.left = contextScope.chunkPositionPercentage + "%";
    chunkMarker.hidden = false;
    chunkMarker.setAttribute(
        "aria-label",
        "Latest chunk starts at token " + formatOptimizerTokenCount(contextScope.chunkStartTokenPosition)
    );
    document.getElementById("optimizer-context-window-start").textContent = "Token 0";
    document.getElementById("optimizer-context-window-end").textContent =
        formatOptimizerTokenCount(contextScope.contextWindowTokenCount) + " token limit";
    document.getElementById("optimizer-context-chunk-position").textContent =
        "Latest chunk starts at " + formatOptimizerTokenCount(contextScope.chunkStartTokenPosition);
}

function renderOptimizerContext(measurementContext) {
    const contextContainer = document.getElementById("optimizer-context");
    contextContainer.replaceChildren();
    if (!measurementContext) {
        appendOptimizerPill(contextContainer, "No execution profile observed", "muted");
        return;
    }
    appendOptimizerPill(
        contextContainer,
        measurementContext.has_restored_prefix ? "Prefix: restored from cache" : "Prefix: processed from scratch"
    );
    appendOptimizerPill(
        contextContainer,
        measurementContext.has_visual_embeddings ? "Input: text and image" : "Input: text only"
    );
    appendOptimizerPill(
        contextContainer,
        measurementContext.is_mtp_active
            ? "Model path: multi-token prediction active"
            : "Model path: target model only"
    );
    appendOptimizerPill(
        contextContainer,
        measurementContext.are_sparse_experts_paged
            ? "Expert access: streamed as needed"
            : "Expert access: fully in memory"
    );
    appendOptimizerPill(
        contextContainer,
        measurementContext.is_prompt_cache_capture_eligible
            ? "Prompt cache: capture eligible"
            : "Prompt cache: capture unavailable"
    );
    if (measurementContext.is_first_chunk_after_restore) {
        appendOptimizerPill(contextContainer, "First chunk after cache restore", "attention");
    }
    if (measurementContext.has_prior_capacity_reduction) {
        appendOptimizerPill(contextContainer, "Earlier chunk reduced by memory", "attention");
    }
}

function appendOptimizerPill(contextContainer, label, pillTone) {
    const contextPill = document.createElement("span");
    contextPill.className = "optimizer-context-pill" + (pillTone ? " " + pillTone : "");
    contextPill.textContent = label;
    contextContainer.appendChild(contextPill);
}

// The measured-rate leader is descriptive evidence for this context only. The
// optimizer still chooses by projected remaining latency, not this simple rank.
function highestObservedThroughputCandidate(candidateMeasurementSummaries) {
    let highestThroughputCandidate = null;
    let highestObservedTokensPerSecond = 0;
    let hasEqualHighestThroughputCandidate = false;
    candidateMeasurementSummaries.forEach((candidateMeasurement) => {
        const observedTokensPerSecond = optimizerObservedTokensPerSecond(candidateMeasurement);
        if (observedTokensPerSecond <= 0) { return; }
        if (observedTokensPerSecond > highestObservedTokensPerSecond) {
            highestThroughputCandidate = candidateMeasurement;
            highestObservedTokensPerSecond = observedTokensPerSecond;
            hasEqualHighestThroughputCandidate = false;
        } else if (observedTokensPerSecond === highestObservedTokensPerSecond) {
            hasEqualHighestThroughputCandidate = true;
        }
    });
    return hasEqualHighestThroughputCandidate ? null : highestThroughputCandidate;
}

function optimizerObservedTokensPerSecond(candidateMeasurement) {
    const measurementCount = Number(candidateMeasurement.measurement_count || 0);
    const averageProcessedTokens = Number(candidateMeasurement.average_processed_prompt_token_count || 0);
    const averageForwardMillis = Number(candidateMeasurement.average_forward_elapsed_millis || 0);
    if (measurementCount === 0 || averageProcessedTokens === 0 || averageForwardMillis === 0) {
        return 0;
    }
    return averageProcessedTokens * 1000 / averageForwardMillis;
}

function renderOptimizerCandidateMeasurements(optimizerDocument, latestChunkOutcome) {
    const comparisonList = document.getElementById("optimizer-measurements-body");
    comparisonList.replaceChildren();
    if (!latestChunkOutcome) {
        const emptyState = document.createElement("p");
        emptyState.className = "optimizer-empty-state";
        emptyState.textContent = "Run a prompt to begin a context-specific chunk-size comparison.";
        comparisonList.appendChild(emptyState);
        return;
    }
    const configuredCandidates = optimizerDocument
        ? optimizerDocument.candidate_chunk_size_token_counts || []
        : [];
    const candidateMeasurementSummaries = latestChunkOutcome
        ? latestChunkOutcome.candidate_measurement_summaries || []
        : [];
    const candidatesToRender = candidateMeasurementSummaries.length > 0
        ? candidateMeasurementSummaries
        : configuredCandidates.map((candidateChunkSizeTokens) => ({
            candidate_chunk_size_tokens: candidateChunkSizeTokens,
            measurement_source: "no_measurements_available",
            measurement_count: 0
        }));
    const highestThroughputCandidate = highestObservedThroughputCandidate(candidatesToRender);
    const highestObservedRate = highestThroughputCandidate
        ? optimizerObservedTokensPerSecond(highestThroughputCandidate)
        : Math.max(0, ...candidatesToRender.map(optimizerObservedTokensPerSecond));
    if (candidatesToRender.length === 0) {
        const emptyState = document.createElement("p");
        emptyState.className = "optimizer-empty-state";
        emptyState.textContent = "No configured chunk sizes are available for comparison.";
        comparisonList.appendChild(emptyState);
        return;
    }
    candidatesToRender.forEach((candidateMeasurement) => {
        comparisonList.appendChild(optimizerCandidateMeasurementCard(
            candidateMeasurement,
            highestThroughputCandidate,
            highestObservedRate
        ));
    });
}

function optimizerCandidateMeasurementCard(
    candidateMeasurement,
    highestThroughputCandidate,
    highestObservedRate
) {
    const candidateCard = document.createElement("article");
    candidateCard.className = "optimizer-candidate-card";
    const candidateHeader = document.createElement("div");
    candidateHeader.className = "optimizer-candidate-header";
    const candidateTitle = document.createElement("strong");
    candidateTitle.textContent = formatOptimizerTokenCount(
        candidateMeasurement.candidate_chunk_size_tokens
    ) + "-token capacity";
    const observedRate = optimizerObservedTokensPerSecond(candidateMeasurement);
    const rateTitle = document.createElement("span");
    rateTitle.textContent = observedRate > 0
        ? Math.round(observedRate).toLocaleString() + " forward tok/s"
        : "Not measured";
    candidateHeader.append(candidateTitle, rateTitle);

    const rateTrack = document.createElement("div");
    rateTrack.className = "optimizer-candidate-rate-track";
    const rateBar = document.createElement("span");
    const relativeRatePercentage = highestObservedRate > 0
        ? Math.max(0, Math.min(100, observedRate * 100 / highestObservedRate))
        : 0;
    rateBar.style.width = relativeRatePercentage + "%";
    rateTrack.appendChild(rateBar);

    const measurementDetail = document.createElement("p");
    const measurementCount = Number(candidateMeasurement.measurement_count || 0);
    if (measurementCount > 0) {
        measurementDetail.textContent = measurementCount.toLocaleString()
            + (measurementCount === 1 ? " measurement" : " measurements")
            + " · average "
            + formatOptimizerTokenCount(candidateMeasurement.average_processed_prompt_token_count)
            + " actual tokens in "
            + formatOptimizerDuration(candidateMeasurement.average_forward_elapsed_millis);
    } else {
        measurementDetail.textContent = "Astronomical has not tried this size for the active context profile yet.";
    }

    const candidateFooter = document.createElement("div");
    candidateFooter.className = "optimizer-candidate-footer";
    const sourceTitle = document.createElement("span");
    sourceTitle.textContent = optimizerMeasurementSourceTitle(candidateMeasurement.measurement_source);
    candidateFooter.appendChild(sourceTitle);
    if (candidateMeasurement === highestThroughputCandidate) {
        const highestRateLabel = document.createElement("span");
        highestRateLabel.className = "optimizer-context-best";
        highestRateLabel.textContent = "Highest forward rate for this profile";
        candidateFooter.appendChild(highestRateLabel);
        candidateCard.dataset.highestRate = "true";
    }
    candidateCard.append(candidateHeader, rateTrack, measurementDetail, candidateFooter);
    return candidateCard;
}

function renderOptimizerLatestDecision(latestChunkOutcome) {
    const decisionPresentation = optimizerDecisionPresentation(latestChunkOutcome);
    document.getElementById("optimizer-latest-headline").textContent = decisionPresentation.headline;
    document.getElementById("optimizer-latest-explanation").textContent =
        decisionPresentation.explanation;
    document.getElementById("optimizer-latest-technical").textContent =
        decisionPresentation.technicalDetail;
}

// Translate internal selection reasons into the concrete user outcome and why
// that choice made sense for this one remaining prompt segment.
function optimizerDecisionPresentation(latestChunkOutcome) {
    if (!latestChunkOutcome) {
        return {
            headline: "No chunk decision observed yet",
            explanation: "Run a prompt to see how Astronomical applies the learned context profile.",
            technicalDetail: "No technical details available."
        };
    }
    const selectedCapacityTokens = Number(
        latestChunkOutcome.selection.selected_candidate_chunk_size_tokens || 0
    );
    const processedPromptTokens = Number(latestChunkOutcome.processed_prompt_token_count || 0);
    const reason = latestChunkOutcome.selection.reason;
    let explanation;
    if (reason === "explore_unmeasured_candidate") {
        explanation = "Astronomical tried this capacity to collect missing evidence for the material execution profile.";
    } else if (reason === "minimize_projected_remaining_prompt_latency") {
        explanation = "Measurements for this context profile predicted this capacity would finish the remaining prompt sooner.";
    } else if (reason === "remaining_tokens_below_smallest_candidate") {
        explanation = "The remaining prompt was smaller than every configured capacity, so Astronomical processed the short tail directly.";
    } else {
        explanation = "Astronomical recorded the decision, but this version does not recognize its selection reason.";
    }
    if (latestChunkOutcome.was_reduced_by_memory_capacity) {
        explanation += " Available model memory reduced the amount actually processed.";
    }
    return {
        headline: "Processed " + formatOptimizerTokenCount(processedPromptTokens)
            + " prompt tokens in " + formatOptimizerDuration(latestChunkOutcome.forward_elapsed_millis),
        explanation,
        technicalDetail: "Selected capacity: " + formatOptimizerTokenCount(selectedCapacityTokens)
            + " tokens · Actual processed: " + formatOptimizerTokenCount(processedPromptTokens)
            + " tokens · Reason: " + optimizerSelectionReasonTitle(reason)
    };
}

function renderOptimizerChunkOutcomes(recentChunkOutcomes) {
    const outcomeList = document.getElementById("optimizer-outcomes");
    outcomeList.replaceChildren();
    if (!recentChunkOutcomes || recentChunkOutcomes.length === 0) {
        const emptyOutcome = document.createElement("li");
        emptyOutcome.textContent = "No decisions observed in this worker session.";
        outcomeList.appendChild(emptyOutcome);
        return;
    }
    recentChunkOutcomes.forEach((chunkOutcome) => {
        const decisionPresentation = optimizerDecisionPresentation(chunkOutcome);
        const outcomeListItem = document.createElement("li");
        const outcomeSummary = document.createElement("strong");
        outcomeSummary.textContent = decisionPresentation.headline;
        const outcomeDetail = document.createElement("span");
        outcomeDetail.textContent = decisionPresentation.explanation;
        outcomeListItem.append(outcomeSummary, outcomeDetail);
        outcomeList.appendChild(outcomeListItem);
    });
}

function optimizerPositionRangeTitle(measurementContext) {
    if (!measurementContext) { return "the next observed token range"; }
    const rangeStart = Number(measurementContext.position_range_start_token_position || 0);
    const rangeEndInclusive = Math.max(
        rangeStart,
        Number(measurementContext.position_range_end_token_position_exclusive || 0) - 1
    );
    return "tokens " + formatOptimizerTokenCount(rangeStart)
        + "–" + formatOptimizerTokenCount(rangeEndInclusive);
}

function optimizerSelectionReasonTitle(selectionReason) {
    if (selectionReason === "explore_unmeasured_candidate") { return "Explore unmeasured candidate"; }
    if (selectionReason === "minimize_projected_remaining_prompt_latency") { return "Minimize projected remaining prompt latency"; }
    if (selectionReason === "remaining_tokens_below_smallest_candidate") { return "Remaining tokens below smallest candidate"; }
    return "Unknown selection reason";
}

function optimizerMeasurementSourceTitle(measurementSource) {
    if (measurementSource === "execution_profile") { return "Evidence for this execution profile"; }
    return "No evidence for this context yet";
}

function formatOptimizerDuration(elapsedMillis) {
    const normalizedMillis = Math.max(0, Number(elapsedMillis || 0));
    if (normalizedMillis < 1000) { return normalizedMillis.toLocaleString() + " ms"; }
    return (normalizedMillis / 1000).toFixed(2).replace(/\.00$/, "") + " seconds";
}

function formatOptimizerTokenCount(tokenCount) {
    return Number(tokenCount || 0).toLocaleString("en-US");
}
