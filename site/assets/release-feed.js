// Self-updating release feed for the public Pages site.
//
// Reads the signed Sparkle appcast (same-origin source of truth at
// ./appcast.xml) and renders:
//   - the current "Latest Stable" version line
//   - a derived "What's New" changelog from the appcast release notes
//
// One fetch, parsed once, shared by every component. The page degrades
// gracefully when the feed is unreachable: the fallback copy (already in the
// HTML) is left untouched and a plain <p> note is appended. No third-party
// API, no build step, nothing to re-pin on the next release.

(function () {
  'use strict';

  var APPCAST_URL = './appcast.xml';
  var MAX_RELEASE_NOTES = 6;

  var feedPromise = null;

  function feed() {
    if (!feedPromise) {
      feedPromise = fetch(APPCAST_URL, { cache: 'no-store' })
        .then(function (response) {
          if (!response.ok) {
            throw new Error('appcast HTTP ' + response.status);
          }
          return response.text();
        })
        .then(parseFeed);
    }
    return feedPromise;
  }

  function parseFeed(xmlText) {
    var documentNode = new DOMParser().parseFromString(xmlText, 'text/xml');
    var itemNodes = documentNode.querySelectorAll('item');
    var releases = [];
    for (var i = 0; i < itemNodes.length; i += 1) {
      releases.push({
        title: firstByLocalName(itemNodes[i], 'title'),
        // shortVersionString is namespaced as sparkle:shortVersionString;
        // match on localName so the prefix never matters.
        version: firstByLocalName(itemNodes[i], 'shortVersionString'),
        date: firstByLocalName(itemNodes[i], 'pubDate'),
        notes: firstByLocalName(itemNodes[i], 'description'),
      });
    }
    return releases;
  }

  function firstByLocalName(node, localName) {
    var found =
      node.querySelector('*[local-name="' + localName + '"]') ||
      Array.prototype.filter
        .call(node.children, function (child) {
          return child.localName === localName;
        })[0];
    return found ? (found.textContent || '').trim() : '';
  }

  function latestVersionLabel(releases) {
    var top = releases[0];
    return top && top.version ? 'v' + top.version : '';
  }

  function renderLatestStable(releases) {
    var versionLabel = latestVersionLabel(releases);
    var versionNode = document.getElementById('release-latest-version');
    var linkNode = document.getElementById('release-latest-link');
    if (!versionLabel || !versionNode) {
      renderFeedFailure('release-latest-failure');
      return;
    }
    versionNode.textContent = versionLabel;
    if (linkNode) {
      linkNode.href =
        'https://github.com/aosama/astronomical/releases/tag/' + versionLabel;
    }
  }

  function renderChangelog(releases) {
    var host = document.getElementById('release-changelog');
    if (!host) {
      return;
    }
    if (!releases.length) {
      host.innerHTML =
        '<p class="release-note-empty">No release notes published yet.</p>';
      return;
    }
    host.innerHTML = releases
      .slice(0, MAX_RELEASE_NOTES)
      .map(function (release) {
        var versionLabel = release.version
          ? 'v' + escapeHtml(release.version)
          : escapeHtml(release.title);
        var dateLine = release.date
          ? '<time class="release-note-date" datetime="' +
            escapeHtml(release.date) +
            '">' +
            escapeHtml(release.date) +
            '</time>'
          : '';
        return (
          '<article class="release-note" role="listitem">' +
          '<h3 class="release-note-title">' +
          versionLabel +
          '</h3>' +
          dateLine +
          '<div class="release-note-body">' +
          renderMarkdown(stripLeadingTitle(release.notes)) +
          '</div>' +
          '</article>'
        );
      })
      .join('');
  }

  function renderFeedFailure() {
    var versionNode = document.getElementById('release-latest-version');
    if (versionNode) {
      versionNode.textContent = 'unavailable right now';
    }
    var changelogHost = document.getElementById('release-changelog');
    if (changelogHost) {
      changelogHost.innerHTML =
        '<p class="release-note-empty">Unable to load the release feed right now — the current release is always available on GitHub Releases.</p>';
    }
  }

  // Safe markdown subset renderer. Input is escaped before any tag is
  // emitted, so descriptions from the appcast can never inject markup.
  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (character) {
      return {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      }[character];
    });
  }

  // The appcast notes open with a "# Astronomical vX.Y.Z" title that the card
  // heading already shows; drop that leading H1 so the card body starts at the
  // first section instead of duplicating the title.
  function stripLeadingTitle(markdown) {
    if (!markdown) {
      return '';
    }
    return markdown.replace(/^\s*#+\s+[^\n]*\n+/, '');
  }

  function renderMarkdown(markdown) {
    if (!markdown || !markdown.trim()) {
      return '';
    }
    var lines = markdown.replace(/\r\n?/g, '\n').split('\n');
    var html = '';
    var inList = false;

    function closeList() {
      if (inList) {
        html += '</ul>';
        inList = false;
      }
    }

    for (var i = 0; i < lines.length; i += 1) {
      var line = escapeHtml(lines[i]);
      if (line.trim() === '') {
        closeList();
        continue;
      }
      var heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        closeList();
        var headingLevel = heading[1].length;
        html +=
          '<h' +
          headingLevel +
          '>' +
          renderInline(heading[2]) +
          '</h' +
          headingLevel +
          '>';
        continue;
      }
      var bullet = line.match(/^\s*[-*]\s+(.*)$/);
      if (bullet) {
        if (!inList) {
          html += '<ul>';
          inList = true;
        }
        html += '<li>' + renderInline(bullet[1]) + '</li>';
        continue;
      }
      closeList();
      html += '<p>' + renderInline(line) + '</p>';
    }
    closeList();
    return html;
  }

  function renderInline(text) {
    return text
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
  }

  function boot() {
    feed()
      .then(function (releases) {
        renderLatestStable(releases);
        renderChangelog(releases);
      })
      .catch(function () {
        renderFeedFailure();
      });
  }

  boot();
})();