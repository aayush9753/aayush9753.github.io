document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('search-input');
  const searchButton = document.getElementById('search-button');
  const searchResults = document.getElementById('search-results');
  const contentTypeCheckboxes = document.querySelectorAll('input[name="content-type"]');
  const sortOrder = document.getElementById('sort-order');
  
  // Get search query from URL if present
  const urlParams = new URLSearchParams(window.location.search);
  const queryParam = urlParams.get('q');
  
  if (queryParam) {
    searchInput.value = queryParam;
    performSearch(queryParam);
  }
  
  // Set up event listeners
  searchButton.addEventListener('click', function() {
    const query = searchInput.value.trim();
    if (query.length > 1) {
      performSearch(query);
      // Update URL with search query
      const newUrl = new URL(window.location);
      newUrl.searchParams.set('q', query);
      window.history.pushState({}, '', newUrl);
    }
  });
  
  searchInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      const query = searchInput.value.trim();
      if (query.length > 1) {
        performSearch(query);
        // Update URL with search query
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('q', query);
        window.history.pushState({}, '', newUrl);
      }
    }
  });
  
  contentTypeCheckboxes.forEach(checkbox => {
    checkbox.addEventListener('change', function() {
      const query = searchInput.value.trim();
      if (query.length > 1) {
        performSearch(query);
      }
    });
  });
  
  sortOrder.addEventListener('change', function() {
    const query = searchInput.value.trim();
    if (query.length > 1) {
      performSearch(query);
    }
  });
  
  function performSearch(query) {
    // Clear previous results
    searchResults.innerHTML = '';
    
    // Get selected content types
    const selectedTypes = Array.from(contentTypeCheckboxes)
      .filter(checkbox => checkbox.checked)
      .map(checkbox => checkbox.value);
    
    if (selectedTypes.length === 0) {
      searchResults.innerHTML = '<p>Please select at least one content type to search.</p>';
      return;
    }
    
    // Normalize query for searching
    const normalizedQuery = query.toLowerCase();
    
    // Collect all results
    let allResults = [];
    
    // Search papers
    if (selectedTypes.includes('papers')) {
      const paperElements = document.querySelectorAll('#papers-data > div');
      paperElements.forEach(paper => {
        const score = calculateRelevance(paper, normalizedQuery);
        if (score > 0) {
          allResults.push({
            type: 'paper',
            title: paper.getAttribute('data-title'),
            date: paper.getAttribute('data-date'),
            authors: paper.getAttribute('data-authors'),
            abstract: paper.getAttribute('data-abstract'),
            tags: paper.getAttribute('data-tags'),
            url: paper.getAttribute('data-url'),
            score: score
          });
        }
      });
    }
    
    // Search notes
    if (selectedTypes.includes('notes')) {
      const noteElements = document.querySelectorAll('#notes-data > div');
      noteElements.forEach(note => {
        const score = calculateRelevance(note, normalizedQuery);
        if (score > 0) {
          allResults.push({
            type: 'note',
            title: note.getAttribute('data-title'),
            date: note.getAttribute('data-date'),
            content: note.getAttribute('data-content'),
            tags: note.getAttribute('data-tags'),
            url: note.getAttribute('data-url'),
            score: score
          });
        }
      });
    }
    
    // Search topics
    if (selectedTypes.includes('topics')) {
      const topicElements = document.querySelectorAll('#topics-data > div');
      topicElements.forEach(topic => {
        const score = calculateRelevance(topic, normalizedQuery);
        if (score > 0) {
          allResults.push({
            type: 'topic',
            title: topic.getAttribute('data-title'),
            description: topic.getAttribute('data-description'),
            content: topic.getAttribute('data-content'),
            tags: topic.getAttribute('data-tags'),
            url: topic.getAttribute('data-url'),
            score: score
          });
        }
      });
    }
    
    // Sort results based on selected sort order
    sortResults(allResults);
    
    // Display results
    if (allResults.length > 0) {
      searchResults.innerHTML = `<h2>Search Results: ${allResults.length} match${allResults.length === 1 ? '' : 'es'}</h2>`;
      
      const resultsList = document.createElement('div');
      resultsList.className = 'search-results-list';
      
      allResults.forEach(result => {
        const resultElement = createResultElement(result);
        resultsList.appendChild(resultElement);
      });
      
      searchResults.appendChild(resultsList);
    } else {
      searchResults.innerHTML = `<h2>No results found for "${query}"</h2>
        <p>Try adjusting your search term or content filters.</p>`;
    }
  }
  
  function calculateRelevance(element, query) {
    let score = 0;
    const title = element.getAttribute('data-title').toLowerCase();
    const content = element.getAttribute('data-content')?.toLowerCase() || '';
    const tags = element.getAttribute('data-tags')?.toLowerCase() || '';
    
    // Title matches are most important
    if (title.includes(query)) {
      score += 10;
      // Exact title match or starts with query
      if (title === query || title.startsWith(query + ' ')) {
        score += 5;
      }
    }
    
    // Tag matches are next most important
    if (tags.includes(query)) {
      score += 7;
      // Direct tag match
      const tagList = tags.split(',').map(t => t.trim());
      if (tagList.includes(query)) {
        score += 3;
      }
    }
    
    // Content matches
    if (content.includes(query)) {
      score += 5;
    }
    
    // Additional data fields based on content type
    if (element.hasAttribute('data-abstract') && element.getAttribute('data-abstract')?.toLowerCase().includes(query)) {
      score += 6;
    }
    
    if (element.hasAttribute('data-authors') && element.getAttribute('data-authors')?.toLowerCase().includes(query)) {
      score += 4;
    }
    
    if (element.hasAttribute('data-description') && element.getAttribute('data-description')?.toLowerCase().includes(query)) {
      score += 6;
    }
    
    return score;
  }
  
  function sortResults(results) {
    const sortValue = sortOrder.value;
    
    switch (sortValue) {
      case 'relevance':
        results.sort((a, b) => b.score - a.score);
        break;
      case 'date-desc':
        results.sort((a, b) => {
          if (!a.date) return 1;
          if (!b.date) return -1;
          return new Date(b.date) - new Date(a.date);
        });
        break;
      case 'date-asc':
        results.sort((a, b) => {
          if (!a.date) return 1;
          if (!b.date) return -1;
          return new Date(a.date) - new Date(b.date);
        });
        break;
      case 'title-asc':
        results.sort((a, b) => a.title.localeCompare(b.title));
        break;
      case 'title-desc':
        results.sort((a, b) => b.title.localeCompare(a.title));
        break;
    }
  }
  
  function createResultElement(result) {
    const resultElement = document.createElement('div');
    resultElement.className = `search-result ${result.type}`;
    
    let resultHtml = `
      <h3><a href="${result.url}">${result.title}</a></h3>
      <div class="result-meta">
        <span class="result-type">${result.type.charAt(0).toUpperCase() + result.type.slice(1)}</span>
    `;
    
    if (result.date) {
      const date = new Date(result.date);
      resultHtml += `<span class="result-date">${date.toLocaleDateString()}</span>`;
    }
    
    if (result.authors) {
      resultHtml += `<span class="result-authors">by ${result.authors}</span>`;
    }
    
    resultHtml += `</div>`;
    
    if (result.abstract) {
      resultHtml += `<p class="result-abstract">${truncateText(result.abstract, 200)}</p>`;
    } else if (result.description) {
      resultHtml += `<p class="result-description">${truncateText(result.description, 200)}</p>`;
    } else if (result.content) {
      resultHtml += `<p class="result-content">${truncateText(result.content, 200)}</p>`;
    }
    
    if (result.tags) {
      const tags = result.tags.split(',').map(tag => tag.trim());
      if (tags.length > 0 && tags[0] !== '') {
        resultHtml += `<div class="result-tags">`;
        tags.forEach(tag => {
          resultHtml += `<span class="tag">${tag}</span>`;
        });
        resultHtml += `</div>`;
      }
    }
    
    resultElement.innerHTML = resultHtml;
    return resultElement;
  }
  
  function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  }
});