---
layout: page
title: Topics
description: Browse research papers by topic
permalink: /topics/
---

<div class="topics-container">
  <h1>Research Topics</h1>
  
  <div class="topics-search">
    <input type="text" id="topic-search" placeholder="Search topics..." />
  </div>
  
  <div class="topics-grid">
    {% assign topics = site.topics | sort: "title" %}
    
    {% if topics.size > 0 %}
      {% for topic in topics %}
      <div class="topic-card" data-title="{{ topic.title | downcase }}">
        <h3><a href="{{ topic.url | relative_url }}">{{ topic.title }}</a></h3>
        
        <p class="topic-description">{{ topic.description | truncate: 120 }}</p>
        
        <div class="topic-meta">
          {% assign topic_papers = site.papers | where_exp: "paper", "paper.topics contains topic.topic_id" %}
          <span class="topic-paper-count">{{ topic_papers.size }} papers</span>
          
          {% if topic.tags.size > 0 %}
          <div class="topic-tags">
            {% for tag in topic.tags limit: 3 %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
            {% if topic.tags.size > 3 %}
            <span class="more-tags">+{{ topic.tags.size | minus: 3 }}</span>
            {% endif %}
          </div>
          {% endif %}
        </div>
        
        {% if topic_papers.size > 0 %}
        <div class="topic-recent-papers">
          <h4>Recent Papers:</h4>
          <ul>
            {% assign recent_papers = topic_papers | sort: "date" | reverse | limit: 3 %}
            {% for paper in recent_papers %}
            <li><a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></li>
            {% endfor %}
          </ul>
        </div>
        {% endif %}
      </div>
      {% endfor %}
    {% else %}
    <p>No topics defined yet.</p>
    {% endif %}
  </div>
</div>

<script src="{{ '/assets/js/topics.js' | relative_url }}"></script>