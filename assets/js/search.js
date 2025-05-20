document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('search-input');
  const searchButton = document.getElementById('search-button');
  const searchResults = document.getElementById('search-results');
  
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
  
  function performSearch(query) {
    // Clear previous results
    searchResults.innerHTML = '';
    
    // Normalize query for searching
    const normalizedQuery = query.toLowerCase();
    
    // Collect all results
    let allResults = [];
    
    // Search papers
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
    
    // Search notes
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
    
    // Search topics
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
    
    // Sort results by relevance score
    allResults.sort((a, b) => b.score - a.score);
    
    // Display results
    if (allResults.length > 0) {
      searchResults.innerHTML = `<h2>Search Results: ${allResults.length} match${allResults.length === 1 ? '' : 'es'}</h2>`;
      
      const resultsList = document.createElement('ul');
      resultsList.className = 'search-results-list';
      
      allResults.forEach(result => {
        const resultElement = createResultElement(result);
        resultsList.appendChild(resultElement);
      });
      
      searchResults.appendChild(resultsList);
    } else {
      searchResults.innerHTML = `<h2>No results found for "${query}"</h2>
        <p>Try adjusting your search term.</p>`;
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
    }
    
    // Tag matches are next most important
    if (tags.includes(query)) {
      score += 7;
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
  
  function createResultElement(result) {
    const resultItem = document.createElement('li');
    
    let resultHtml = '';
    
    if (result.date) {
      resultHtml += `<span class="date">${formatDate(result.date)}</span> `;
    }
    
    resultHtml += `<a href="${result.url}">${result.title}</a>`;
    
    if (result.authors) {
      resultHtml += ` <span class="authors">${result.authors}</span>`;
    }
    
    if (result.tags) {
      const tags = result.tags.split(',').map(tag => tag.trim());
      if (tags.length > 0 && tags[0] !== '') {
        resultHtml += `<div class="tags">Tags: `;
        tags.forEach((tag, index) => {
          resultHtml += `<span class="tag"><a href="/search/?q=${encodeURIComponent(tag)}">${tag}</a></span>`;
        });
        resultHtml += `</div>`;
      }
    }
    
    resultItem.innerHTML = resultHtml;
    return resultItem;
  }
  
  function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toISOString().split('T')[0]; // YYYY-MM-DD format
  }
});