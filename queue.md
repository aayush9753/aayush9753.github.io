---
layout: page
title: Reading Queue
description: Papers and articles to read
permalink: /queue/
---

<div class="reading-queue-container">
  <h1>Reading Queue</h1>
  
  <div class="queue-filters">
    <label>Filter by Priority:</label>
    <button class="priority-filter active" data-priority="all">All</button>
    <button class="priority-filter" data-priority="high">High</button>
    <button class="priority-filter" data-priority="medium">Medium</button>
    <button class="priority-filter" data-priority="low">Low</button>
    
    <label>Sort by:</label>
    <select id="queue-sort">
      <option value="priority">Priority</option>
      <option value="date-added">Date Added</option>
      <option value="title">Title</option>
    </select>
  </div>
  
  <div class="queue-list">
    {% assign queue_papers = site.papers | where: "queue", true | sort: "priority" %}
    
    {% if queue_papers.size > 0 %}
    <div class="papers-list queue">
      {% for paper in queue_papers %}
      <div class="paper-card queue priority-{{ paper.priority | downcase }}">
        <div class="paper-priority priority-{{ paper.priority | downcase }}">
          {{ paper.priority | capitalize }}
        </div>
        
        <h3><a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></h3>
        
        <p class="paper-meta">
          <span class="paper-authors">{{ paper.authors | join: ", " }}</span>
          <span class="paper-date">{{ paper.date | date: "%Y" }}</span>
          <span class="paper-pages">{{ paper.pages }} pages</span>
          {% if paper.reading_time %}
          <span class="paper-reading-time">~{{ paper.reading_time }} min read</span>
          {% endif %}
        </p>
        
        <p class="paper-abstract">{{ paper.abstract | truncate: 150 }}</p>
        
        <div class="paper-tags">
          {% for tag in paper.tags %}
          <span class="tag">{{ tag }}</span>
          {% endfor %}
        </div>
        
        <div class="queue-actions">
          <span class="added-date">Added on {{ paper.queue_date | date: "%b %d, %Y" }}</span>
          <a href="{{ paper.url | relative_url }}" class="start-reading-btn">Start Reading</a>
        </div>
      </div>
      {% endfor %}
    </div>
    {% else %}
    <p>Your reading queue is empty.</p>
    {% endif %}
  </div>
</div>

<script src="{{ '/assets/js/queue.js' | relative_url }}"></script>