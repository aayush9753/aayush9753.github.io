---
layout: default
title: Reading Queue
description: Papers and articles to read
permalink: /queue/
---

# Reading Queue

{% assign queue_papers = site.papers | where: "queue", true | sort: "priority" %}
{% if queue_papers.size > 0 %}
<ul class="queue-list">
  {% for paper in queue_papers %}
  <li>
    <span class="date">{{ paper.queue_date | date: "%Y-%m-%d" }}</span>
    <a href="{{ paper.url | relative_url }}">{{ paper.title }}</a>
    <span class="priority">({{ paper.priority }} priority)</span>
    {% if paper.tags.size > 0 %}
    <div class="tags">
      Tags: 
      {% for tag in paper.tags %}
      <span class="tag"><a href="/search/?q={{ tag | uri_escape }}">{{ tag }}</a></span>
      {% endfor %}
    </div>
    {% endif %}
  </li>
  {% endfor %}
</ul>
{% else %}
<p>Your reading queue is empty.</p>
{% endif %}