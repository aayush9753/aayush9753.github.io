---
layout: page
title: Research Papers
description: Browse all research papers
permalink: /papers/
---

<div class="papers-container">
  <h1>Research Papers</h1>
  
  <div class="papers-filters">
    <div class="filter-group">
      <label>Filter by Topic:</label>
      <select id="topic-filter">
        <option value="all">All Topics</option>
        {% assign topics = site.topics | sort: "title" %}
        {% for topic in topics %}
        <option value="{{ topic.topic_id }}">{{ topic.title }}</option>
        {% endfor %}
      </select>
    </div>
    
    <div class="filter-group">
      <label>Filter by Tag:</label>
      <div class="tag-filters">
        {% assign all_tags = "" | split: "" %}
        {% for paper in site.papers %}
          {% assign all_tags = all_tags | concat: paper.tags %}
        {% endfor %}
        
        {% assign unique_tags = all_tags | uniq | sort %}
        {% for tag in unique_tags limit: 15 %}
        <button class="tag-filter" data-tag="{{ tag }}">{{ tag }}</button>
        {% endfor %}
        
        {% if unique_tags.size > 15 %}
        <div class="tag-filter-more">
          <button id="show-more-tags">More tags...</button>
          <div class="more-tags-dropdown">
            {% for tag in unique_tags offset: 15 %}
            <button class="tag-filter" data-tag="{{ tag }}">{{ tag }}</button>
            {% endfor %}
          </div>
        </div>
        {% endif %}
      </div>
    </div>
    
    <div class="filter-group">
      <label>Sort by:</label>
      <select id="papers-sort">
        <option value="date-desc">Date (Newest)</option>
        <option value="date-asc">Date (Oldest)</option>
        <option value="title-asc">Title (A-Z)</option>
        <option value="title-desc">Title (Z-A)</option>
      </select>
    </div>
  </div>
  
  <div class="papers-list-container">
    <div class="active-filters">
      <span class="active-filter-label">Active Filters:</span>
      <div id="active-filters-list">
        <span class="no-filters">None</span>
      </div>
      <button id="clear-filters">Clear All</button>
    </div>
    
    <div class="papers-list">
      {% assign all_papers = site.papers | sort: "date" | reverse %}
      
      {% if all_papers.size > 0 %}
        {% for paper in all_papers %}
        <div class="paper-card" 
          data-topics="{{ paper.topics | join: ',' }}"
          data-tags="{{ paper.tags | join: ',' }}"
          data-date="{{ paper.date | date: '%Y-%m-%d' }}"
          data-title="{{ paper.title }}">
          
          <h3><a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></h3>
          
          <p class="paper-meta">
            <span class="paper-authors">{{ paper.authors | join: ", " }}</span>
            <span class="paper-date">{{ paper.date | date: "%Y" }}</span>
            {% if paper.publication %}
            <span class="paper-publication">{{ paper.publication }}</span>
            {% endif %}
          </p>
          
          <p class="paper-abstract">{{ paper.abstract | truncate: 200 }}</p>
          
          <div class="paper-footer">
            <div class="paper-topics">
              {% for topic_id in paper.topics %}
                {% assign topic = site.topics | where: "topic_id", topic_id | first %}
                {% if topic %}
                <a href="{{ topic.url | relative_url }}" class="paper-topic">{{ topic.title }}</a>
                {% endif %}
              {% endfor %}
            </div>
            
            <div class="paper-tags">
              {% for tag in paper.tags %}
              <span class="tag">{{ tag }}</span>
              {% endfor %}
            </div>
          </div>
        </div>
        {% endfor %}
      {% else %}
      <p>No papers added yet.</p>
      {% endif %}
    </div>
  </div>
</div>

<script src="{{ '/assets/js/papers.js' | relative_url }}"></script>