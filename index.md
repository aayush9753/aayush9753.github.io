---
layout: home
title: Research Reading Library
description: A personal collection of research papers, articles, and reading notes
---

# Research Reading Library

Welcome to my personal research library - a curated collection of academic papers, research articles, and personal notes organized for easy reference and discovery.

## Recent Additions

{% assign recent_papers = site.papers | sort: "date" | reverse | limit: 5 %}
{% if recent_papers.size > 0 %}
<div class="recent-papers">
  {% for paper in recent_papers %}
  <div class="paper-card">
    <h3><a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></h3>
    <p class="paper-meta">
      <span class="paper-authors">{{ paper.authors | join: ", " }}</span>
      <span class="paper-date">{{ paper.date | date: "%Y" }}</span>
    </p>
    <p class="paper-abstract">{{ paper.abstract | truncate: 150 }}</p>
    <div class="paper-tags">
      {% for tag in paper.tags %}
      <span class="tag">{{ tag }}</span>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</div>
{% else %}
<p>No papers added yet. Get started by adding your first research paper!</p>
{% endif %}

## Reading Queue

{% assign queue_papers = site.papers | where: "queue", true | sort: "priority" | limit: 3 %}
{% if queue_papers.size > 0 %}
<div class="queue-papers">
  {% for paper in queue_papers %}
  <div class="paper-card queue">
    <h3><a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></h3>
    <p class="paper-meta">
      <span class="paper-authors">{{ paper.authors | join: ", " }}</span>
      <span class="paper-priority">Priority: {{ paper.priority }}</span>
    </p>
    <div class="paper-tags">
      {% for tag in paper.tags %}
      <span class="tag">{{ tag }}</span>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</div>
{% else %}
<p>Your reading queue is empty.</p>
{% endif %}

## Browse by Topic

{% assign topics = site.topics | sort: "title" %}
{% if topics.size > 0 %}
<div class="topics-grid">
  {% for topic in topics %}
  <div class="topic-card">
    <h3><a href="{{ topic.url | relative_url }}">{{ topic.title }}</a></h3>
    <p>{{ topic.description | truncate: 100 }}</p>
    <p class="topic-count">{{ topic.paper_count }} papers</p>
  </div>
  {% endfor %}
</div>
{% else %}
<p>No topics defined yet. Start by creating topic categories for your research papers.</p>
{% endif %}

[View All Papers](/papers/) | [View Reading Queue](/queue/) | [Search](/search/)