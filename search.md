---
layout: default
title: Search
description: Search through research papers, notes, and topics
permalink: /search/
---

# Search

<div class="search-form">
  <input type="text" id="search-input" placeholder="Search papers, notes, and topics..." />
  <button id="search-button">Search</button>
</div>

<div id="search-results">
  <!-- Results will be inserted here by JavaScript -->
  <p class="search-instructions">Enter a search term to find papers, notes, and topics.</p>
</div>

<div id="search-data" style="display: none;">
  {% assign all_papers = site.papers | sort: "date" | reverse %}
  {% assign all_notes = site.notes | sort: "date" | reverse %}
  {% assign all_topics = site.topics | sort: "title" %}
  
  <div id="papers-data">
    {% for paper in all_papers %}
    <div 
      data-title="{{ paper.title }}" 
      data-authors="{{ paper.authors | join: ', ' }}" 
      data-date="{{ paper.date | date: "%Y-%m-%d" }}" 
      data-abstract="{{ paper.abstract }}" 
      data-content="{{ paper.content | strip_html }}" 
      data-tags="{{ paper.tags | join: ', ' }}"
      data-type="paper"
      data-url="{{ paper.url | relative_url }}"
    ></div>
    {% endfor %}
  </div>
  
  <div id="notes-data">
    {% for note in all_notes %}
    <div 
      data-title="{{ note.title }}" 
      data-date="{{ note.date | date: "%Y-%m-%d" }}" 
      data-content="{{ note.content | strip_html }}" 
      data-tags="{{ note.tags | join: ', ' }}"
      data-type="note"
      data-url="{{ note.url | relative_url }}"
    ></div>
    {% endfor %}
  </div>
  
  <div id="topics-data">
    {% for topic in all_topics %}
    <div 
      data-title="{{ topic.title }}" 
      data-description="{{ topic.description }}" 
      data-content="{{ topic.content | strip_html }}" 
      data-tags="{{ topic.tags | join: ', ' }}"
      data-type="topic"
      data-url="{{ topic.url | relative_url }}"
    ></div>
    {% endfor %}
  </div>
</div>

<script src="{{ '/assets/js/search.js' | relative_url }}"></script>