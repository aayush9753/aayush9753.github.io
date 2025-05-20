---
layout: default
title: Research Papers
description: Browse all research papers
permalink: /papers/
---

# Research Papers

## Filter by Tag

{% assign all_tags = "" | split: "" %}
{% for paper in site.papers %}
  {% assign all_tags = all_tags | concat: paper.tags %}
{% endfor %}

{% assign unique_tags = all_tags | uniq | sort %}
{% for tag in unique_tags %}
<a href="/search/?q={{ tag | uri_escape }}">{{ tag }}</a>{% unless forloop.last %} | {% endunless %}
{% endfor %}

## All Papers

{% assign all_papers = site.papers | sort: "date" | reverse %}
{% if all_papers.size > 0 %}
<ul class="papers-list">
  {% for paper in all_papers %}
  <li>
    <span class="date">{{ paper.date | date: "%Y-%m-%d" }}</span>
    <a href="{{ paper.url | relative_url }}">{{ paper.title }}</a>
    <span class="authors">{{ paper.authors | first }} et al.</span>
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

{% if all_papers.size > 20 %}
<div class="older-posts">
  <a href="/papers/archive/">Older posts...</a>
</div>
{% endif %}

{% else %}
<p>No papers added yet.</p>
{% endif %}