---
layout: home
title: Research Reading Library
description: A personal collection of research papers, articles, and reading notes
---

# Research Reading Library

A personal collection of research papers, articles, and reading notes.

## Recent Papers

{% assign recent_papers = site.papers | sort: "date" | reverse | limit: 5 %}
{% if recent_papers.size > 0 %}
<ul>
{% for paper in recent_papers %}
<li>{{ paper.date | date: "%Y-%m-%d" }} <a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></li>
{% endfor %}
</ul>
{% else %}
No papers added yet. Get started by adding your first research paper!
{% endif %}

## Reading Queue

{% assign queue_papers = site.papers | where: "queue", true | sort: "priority" | limit: 3 %}
{% if queue_papers.size > 0 %}
<ul>
{% for paper in queue_papers %}
<li>{{ paper.queue_date | date: "%Y-%m-%d" }} <a href="{{ paper.url | relative_url }}">{{ paper.title }}</a> ({{ paper.priority }} priority)</li>
{% endfor %}
</ul>
{% else %}
Your reading queue is empty.
{% endif %}

## Topics

{% assign topics = site.topics | sort: "title" %}
{% if topics.size > 0 %}
<ul>
{% for topic in topics %}
<li><a href="{{ topic.url | relative_url }}">{{ topic.title }}</a>
    <ul>
    {% assign topic_papers = site.papers | where_exp: "paper", "paper.topics contains topic.topic_id" | sort: "date" | reverse | limit: 3 %}
    {% for paper in topic_papers %}
    <li>{{ paper.date | date: "%Y-%m-%d" }} <a href="{{ paper.url | relative_url }}">{{ paper.title }}</a></li>
    {% endfor %}
    </ul>
</li>
{% endfor %}
</ul>
{% else %}
No topics defined yet. Start by creating topic categories for your research papers.
{% endif %}

## Links

* [View All Papers](/papers/)
* [View Reading Queue](/queue/)
* [Search](/search/)