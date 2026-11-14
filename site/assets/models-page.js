// The Models page: lists the models Astronomical has checked, each with an
// "Open in Astronomical" link.
//
// The list is read from the catalog file in the project's repository, which
// is the same file the app is built from, so the page never keeps a copy
// that could fall behind. When that file cannot be read, the page says so
// and points at the app's own Models page; it shows no stale list.

(function () {
  'use strict';

  var CATALOG_URL =
    'https://raw.githubusercontent.com/aosama/astronomical/main/registry/download_catalog.json';
  var OPEN_LINK_PREFIX = 'astronomical://models/huggingface/';
  var HUGGING_FACE_PREFIX = 'https://huggingface.co/';
  // An owner and a name in Hugging Face's own characters. Anything else is
  // left out, so the page never builds a link from a value it does not
  // recognise as a model name.
  var MODEL_ID_PATTERN = /^[A-Za-z0-9._-]{1,96}\/[A-Za-z0-9._-]{1,96}$/;
  var BYTES_PER_GIGABYTE = 1024 * 1024 * 1024;

  // The order and plain names of the groups; a role missing from this list
  // is not shown.
  var MODEL_GROUPS = [
    { title: 'Chat', lede: 'Talk, write, and ask about photos.', roles: ['chat'] },
    { title: 'Coding', lede: 'Help writing, reading, and fixing code.', roles: ['coding'] },
    { title: 'Pictures', lede: 'Turn a written description into a picture.', roles: ['image_generation'] },
    { title: 'Speech', lede: 'Read text aloud.', roles: ['speech'] },
    { title: 'Helpers', lede: 'Small models that search your text or answer set questions.', roles: ['helper', 'decision'] }
  ];

  function element(tagName, className, text) {
    var created = document.createElement(tagName);
    if (className) created.className = className;
    if (text) created.textContent = text;
    return created;
  }

  // Experimental builds and models about to leave the list are kept off this
  // page: a newcomer arriving from a link should meet the dependable ones.
  function isListedModel(catalogEntry) {
    return catalogEntry
      && catalogEntry.public === true
      && catalogEntry.experimental !== true
      && catalogEntry.retiring !== true
      && typeof catalogEntry.huggingface_id === 'string'
      && MODEL_ID_PATTERN.test(catalogEntry.huggingface_id);
  }

  // The same binary gigabytes the app shows, so the page and the app agree.
  function formatSize(sizeBytes) {
    if (typeof sizeBytes !== 'number' || !(sizeBytes > 0)) return '';
    return (sizeBytes / BYTES_PER_GIGABYTE).toFixed(1) + ' GB on disk';
  }

  function modelRow(catalogEntry) {
    var huggingfaceId = catalogEntry.huggingface_id;
    var row = element('li', 'model-row');
    var identity = element('div', 'model-identity');
    identity.appendChild(element('p', 'model-name', huggingfaceId.split('/')[1]));
    if (typeof catalogEntry.summary === 'string') {
      identity.appendChild(element('p', 'model-summary', catalogEntry.summary));
    }
    var facts = element('p', 'model-facts');
    var sizeText = formatSize(catalogEntry.approximate_size_bytes);
    if (sizeText) facts.appendChild(document.createTextNode(sizeText + ' · '));
    var sourceLink = element('a', '', 'View on Hugging Face');
    sourceLink.href = HUGGING_FACE_PREFIX + huggingfaceId;
    sourceLink.rel = 'noopener';
    facts.appendChild(sourceLink);
    identity.appendChild(facts);

    var openLink = element('a', 'button button-primary model-open', 'Open in Astronomical');
    openLink.href = OPEN_LINK_PREFIX + huggingfaceId;
    openLink.setAttribute('aria-label', 'Open ' + huggingfaceId.split('/')[1] + ' in Astronomical');
    row.appendChild(identity);
    row.appendChild(openLink);
    return row;
  }

  function bySize(firstEntry, secondEntry) {
    return (firstEntry.approximate_size_bytes || 0) - (secondEntry.approximate_size_bytes || 0);
  }

  function renderCatalog(catalogContainer, catalogDocument) {
    var entries = Array.isArray(catalogDocument && catalogDocument.entries) ? catalogDocument.entries : [];
    var listedModels = entries.filter(isListedModel);
    if (listedModels.length === 0) {
      throw new Error('the catalog lists no models');
    }
    var sections = [];
    MODEL_GROUPS.forEach(function (modelGroup) {
      var groupModels = listedModels
        .filter(function (catalogEntry) { return modelGroup.roles.indexOf(catalogEntry.role) !== -1; })
        .sort(bySize);
      if (groupModels.length === 0) return;
      sections.push(element('h2', '', modelGroup.title));
      sections.push(element('p', '', modelGroup.lede));
      var list = element('ul', 'model-list');
      groupModels.forEach(function (catalogEntry) { list.appendChild(modelRow(catalogEntry)); });
      sections.push(list);
    });
    catalogContainer.replaceChildren.apply(catalogContainer, sections);
  }

  function renderUnavailable(catalogContainer) {
    var notice = element('p', '', 'The list of models could not be loaded right now. In Astronomical, choose ');
    notice.appendChild(element('strong', '', 'Open Observatory'));
    notice.appendChild(document.createTextNode(' and go to '));
    notice.appendChild(element('strong', '', 'Models'));
    notice.appendChild(document.createTextNode(' to see them all.'));
    catalogContainer.replaceChildren(notice);
  }

  function loadCatalog() {
    var catalogContainer = document.getElementById('model-catalog');
    if (!catalogContainer) return;
    fetch(CATALOG_URL)
      .then(function (response) {
        if (!response.ok) throw new Error('catalog HTTP ' + response.status);
        return response.json();
      })
      .then(function (catalogDocument) { renderCatalog(catalogContainer, catalogDocument); })
      .catch(function () { renderUnavailable(catalogContainer); });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadCatalog);
  } else {
    loadCatalog();
  }
})();
