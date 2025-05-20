# Research Reading Library

A Jekyll-based GitHub Pages website that serves as a personal research library and note-taking system. This website organizes research papers, articles, and reading notes in a clean, searchable, and easily navigable format.

## Features

- **Responsive, minimalist design** optimized for reading and finding research content
- **Logical folder structure** for organizing markdown files by category, topic, and publication date
- **Robust tagging system** for easy cross-referencing of papers and concepts
- **Full-text search functionality** across all content
- **Citation generation** in multiple formats (APA, MLA, Chicago)
- **Dashboard view** showing recent additions and reading progress
- **Reading queue management** with priority levels and tracking

## Getting Started

### Prerequisites

- [Ruby](https://www.ruby-lang.org/en/downloads/) (version 2.7.0 or higher recommended)
- [Bundler](https://bundler.io/) (`gem install bundler`)
- [Jekyll](https://jekyllrb.com/) (`gem install jekyll`)

### Local Development

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/research-reading-library.git
   cd research-reading-library
   ```

2. Install dependencies:
   ```
   bundle install
   ```

3. Run the local development server:
   ```
   bundle exec jekyll serve
   ```

4. Visit `http://localhost:4000` in your browser

## Content Structure

### Adding Papers

Create a new markdown file in the `_papers` directory with the following front matter:

```yaml
---
layout: paper
title: "Paper Title"
authors: ["Author One", "Author Two"]
date: YYYY-MM-DD
publication: "Journal or Conference Name"
paper_id: unique-paper-id
doi: "DOI number"
url_paper: "URL to the paper"
topics: [topic-id-1, topic-id-2]
tags: [tag1, tag2, tag3]
reading_time: 60
progress: 75
queue: true  # if this paper is in your reading queue
priority: "high"  # high, medium, or low (for reading queue)
queue_date: YYYY-MM-DD
abstract: |
  The paper abstract goes here...
key_findings: |
  - Key finding 1
  - Key finding 2
  - Key finding 3
citation:
  apa: "APA citation format"
  mla: "MLA citation format"
  chicago: "Chicago citation format"
related_papers: [paper-id-1, paper-id-2]
---

Your notes and commentary on the paper go here...
```

### Adding Notes

Create a new markdown file in the `_notes` directory with the following front matter:

```yaml
---
layout: note
title: "Note Title"
date: YYYY-MM-DD
last_modified: YYYY-MM-DD
note_id: unique-note-id
tags: [tag1, tag2, tag3]
related_papers: [paper-id-1, paper-id-2]
related_notes: [note-id-1, note-id-2]
---

Your note content goes here...
```

### Adding Topics

Create a new markdown file in the `_topics` directory with the following front matter:

```yaml
---
layout: topic
title: "Topic Title"
description: "Brief description of the topic"
topic_id: unique-topic-id
tags: [tag1, tag2, tag3]
related_topics: [topic-id-1, topic-id-2]
---

Your topic description and overview goes here...
```

## Customization

### Site Configuration

The main configuration settings can be found in `_config.yml`. Modify this file to change:

- Site title and description
- Theme settings and appearance
- Collection configurations
- Plugin settings

### Styling

Custom styles can be added or modified in:

- `assets/css/custom.css` - Main stylesheet for custom styles

### JavaScript

Custom scripts for interactive features are in:

- `assets/js/search.js` - Search functionality
- `assets/js/paper.js` - Paper page features (citations, progress tracking)
- `assets/js/papers.js` - Papers list filtering and sorting
- `assets/js/queue.js` - Reading queue management
- `assets/js/topics.js` - Topics filtering and navigation

## Deployment

This site is designed to be deployed to GitHub Pages:

1. Push your changes to the `main` branch of your GitHub repository
2. GitHub will automatically build and deploy your site
3. Your site will be available at `https://yourusername.github.io/repository-name/`

For custom domain setup, see the [GitHub Pages documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Jekyll](https://jekyllrb.com/)
- Using [Hydejack](https://hydejack.com/) theme
- Search functionality built with JavaScript
- Citation management using custom JavaScript