// Astronomical Observatory Library catalog behavior.
// The release catalog is immutable, so one startup fetch owns all Library rendering.
// Catalog values enter the document only through textContent and created elements.

const LIBRARY_CATALOG_URL = "/v1/library/catalog";
// A local immutable endpoint should resolve promptly; the bound prevents a stale loading state
// when a daemon connection is interrupted without producing an immediate fetch error.
const LIBRARY_CATALOG_TIMEOUT_MILLIS = 10_000;

let libraryCatalogLoadPromise = null;

function wireLibraryCatalog() {
    if (libraryCatalogLoadPromise !== null) {
        return libraryCatalogLoadPromise;
    }
    const catalogContainer = document.getElementById("library-catalog");
    if (!catalogContainer) {
        return Promise.resolve("unavailable");
    }
    libraryCatalogLoadPromise = loadLibraryCatalog(catalogContainer);
    return libraryCatalogLoadPromise;
}

async function loadLibraryCatalog(
    catalogContainer,
    catalogTimeoutMillis = LIBRARY_CATALOG_TIMEOUT_MILLIS
) {
    const catalogAbortController = new AbortController();
    const catalogTimeout = setTimeout(
        () => catalogAbortController.abort(),
        catalogTimeoutMillis
    );
    try {
        const catalogResponse = await fetch(
            LIBRARY_CATALOG_URL,
            { signal: catalogAbortController.signal }
        );
        if (!catalogResponse.ok) {
            return renderLibraryUnavailableState(catalogContainer);
        }
        const catalogDocument = await catalogResponse.json();
        return renderLibraryCatalogDocument(catalogDocument, catalogContainer);
    } catch {
        return renderLibraryUnavailableState(catalogContainer);
    } finally {
        clearTimeout(catalogTimeout);
    }
}

function renderLibraryCatalogDocument(catalogDocument, catalogContainer) {
    const resolvedCatalogContainer = catalogContainer
        || document.getElementById("library-catalog");
    if (!resolvedCatalogContainer) {
        return "unavailable";
    }
    const catalogRows = libraryCatalogRowsFromDocument(catalogDocument);
    if (catalogRows === null) {
        return renderLibraryUnavailableState(resolvedCatalogContainer);
    }
    if (catalogRows.length === 0) {
        return renderLibraryEmptyState(resolvedCatalogContainer);
    }

    const catalogList = document.createElement("div");
    catalogList.className = "library-catalog-list";
    catalogList.replaceChildren(...catalogRows.map(createLibraryCatalogRow));
    const statusMessage = resolveLibraryStatusMessage();
    statusMessage.className = "library-catalog-state library-catalog-status-only";
    statusMessage.textContent = "Model catalog loaded.";
    resolvedCatalogContainer.dataset.libraryState = "ready";
    resolvedCatalogContainer.replaceChildren(statusMessage, catalogList);
    return "ready";
}

function libraryCatalogRowsFromDocument(catalogDocument) {
    if (!catalogDocument
        || catalogDocument.schema_version !== 1
        || !Array.isArray(catalogDocument.entries)) {
        return null;
    }
    const catalogRows = [];
    for (const catalogEntry of catalogDocument.entries) {
        if (!isRenderableLibraryCatalogEntry(catalogEntry)) {
            return null;
        }
        catalogRows.push({
            displayName: catalogEntry.display_name,
            family: catalogEntry.family,
            approximateSize: formatLibrarySizeGigabytes(catalogEntry.approximate_size_bytes)
        });
    }
    return catalogRows;
}

function isRenderableLibraryCatalogEntry(catalogEntry) {
    return catalogEntry !== null
        && typeof catalogEntry === "object"
        && typeof catalogEntry.huggingface_id === "string"
        && catalogEntry.huggingface_id.trim().length > 0
        && typeof catalogEntry.revision === "string"
        && /^[0-9a-f]{40}$/i.test(catalogEntry.revision)
        && typeof catalogEntry.display_name === "string"
        && catalogEntry.display_name.trim().length > 0
        && (catalogEntry.family === "qwen3_5" || catalogEntry.family === "laguna")
        && Number.isSafeInteger(catalogEntry.approximate_size_bytes)
        && catalogEntry.approximate_size_bytes > 0
        && catalogEntry.public === true;
}

function formatLibrarySizeGigabytes(approximateSizeBytes) {
    return (approximateSizeBytes / 1_000_000_000).toFixed(2) + " GB";
}

function createLibraryCatalogRow(catalogRow) {
    const row = document.createElement("article");
    row.className = "library-catalog-row";

    const displayName = document.createElement("h3");
    displayName.textContent = catalogRow.displayName;

    const metadata = document.createElement("dl");
    metadata.className = "library-catalog-metadata";
    const familyLabel = document.createElement("dt");
    familyLabel.textContent = "Family";
    const familyValue = document.createElement("dd");
    familyValue.textContent = catalogRow.family;
    const sizeLabel = document.createElement("dt");
    sizeLabel.textContent = "Approximate size";
    const sizeValue = document.createElement("dd");
    sizeValue.textContent = catalogRow.approximateSize;
    metadata.replaceChildren(familyLabel, familyValue, sizeLabel, sizeValue);

    row.replaceChildren(displayName, metadata);
    return row;
}

function renderLibraryEmptyState(catalogContainer) {
    return renderLibraryStatusState(
        catalogContainer,
        "empty",
        "No catalog entries are available in this release."
    );
}

function renderLibraryUnavailableState(catalogContainer) {
    return renderLibraryStatusState(
        catalogContainer,
        "unavailable",
        "The model catalog is unavailable."
    );
}

function renderLibraryStatusState(catalogContainer, libraryState, message) {
    const statusMessage = resolveLibraryStatusMessage();
    statusMessage.className = "library-catalog-state";
    catalogContainer.dataset.libraryState = libraryState;
    // Preserve one live-region node so assistive technology observes state transitions reliably.
    catalogContainer.replaceChildren(statusMessage);
    statusMessage.textContent = message;
    return libraryState;
}

function resolveLibraryStatusMessage() {
    const existingStatusMessage = document.getElementById("library-catalog-status");
    if (existingStatusMessage) {
        return existingStatusMessage;
    }
    const statusMessage = document.createElement("p");
    statusMessage.id = "library-catalog-status";
    statusMessage.setAttribute("role", "status");
    statusMessage.setAttribute("aria-live", "polite");
    return statusMessage;
}
