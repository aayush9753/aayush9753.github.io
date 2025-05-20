/**
 * Search Functionality for My Second Brain
 * Provides client-side search across posts
 */

class SearchEngine {
  constructor() {
    this.searchable = false;
  }
  
  /**
   * Initialize search engine
   */
  init() {
    // Check if dataStorage is available
    if (typeof dataStorage !== 'undefined') {
      this.searchable = true;
      
      // Set up search UI if it exists
      this.setupSearchUI();
    } else {
      console.error('Search requires dataStorage module');
    }
  }
  
  /**
   * Set up search functionality in the UI
   */
  setupSearchUI() {
    const searchContainer = document.querySelector('.search-container');
    
    // If there's no search container yet, create it
    if (!searchContainer && document.querySelector('header')) {
      const header = document.querySelector('header');
      const searchHTML = `
        <div class="search-container">
          <input type="text" id="search-input" placeholder="Search posts..." aria-label="Search posts">
          <div id="search-results" class="search-results hidden"></div>
        </div>
      `;
      
      header.insertAdjacentHTML('beforeend', searchHTML);
      
      // Add CSS for the search elements
      const style = document.createElement('style');
      style.textContent = `
        .search-container {
          margin-top: var(--spacing-unit);
          position: relative;
        }
        
        #search-input {
          width: 100%;
          padding: calc(var(--spacing-unit) * 0.5);
          border: 1px solid var(--border-color);
          border-radius: 3px;
          font-family: inherit;
          font-size: var(--base-size);
        }
        
        .search-results {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          background-color: white;
          border: 1px solid var(--border-color);
          border-radius: 3px;
          margin-top: 5px;
          max-height: 300px;
          overflow-y: auto;
          z-index: 100;
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .search-result-item {
          padding: calc(var(--spacing-unit) * 0.5);
          border-bottom: 1px solid var(--border-color);
        }
        
        .search-result-item:last-child {
          border-bottom: none;
        }
        
        .search-result-item:hover {
          background-color: var(--subtle-bg);
        }
        
        .search-result-title {
          font-weight: 500;
        }
        
        .search-result-meta {
          font-size: calc(var(--base-size) * 0.8);
          color: var(--secondary-text);
        }
        
        .search-result-snippet {
          font-size: calc(var(--base-size) * 0.9);
          margin-top: calc(var(--spacing-unit) * 0.25);
        }
        
        .search-no-results {
          padding: calc(var(--spacing-unit) * 0.5);
          color: var(--secondary-text);
          text-align: center;
        }
        
        .hidden {
          display: none;
        }
      `;
      document.head.appendChild(style);
      
      // Add event listener to the search input
      const searchInput = document.getElementById('search-input');
      const searchResults = document.getElementById('search-results');
      
      searchInput.addEventListener('input', () => {
        const query = searchInput.value.trim().toLowerCase();
        
        if (query.length < 2) {
          searchResults.classList.add('hidden');
          return;
        }
        
        const results = this.search(query);
        this.displayResults(results, searchResults);
      });
      
      // Close search results when clicking outside
      document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
          searchResults.classList.add('hidden');
        }
      });
    }
  }
  
  /**
   * Perform search against the index
   */
  search(query) {
    if (!this.searchable) return [];
    
    // Get posts from storage
    const posts = dataStorage.getPosts();
    
    const terms = query.toLowerCase().split(' ');
    return posts
      .filter(item => {
        // Check if all terms are found somewhere in the item
        return terms.every(term => 
          item.title.toLowerCase().includes(term) ||
          item.content.toLowerCase().includes(term) ||
          item.tags.some(tag => tag.toLowerCase().includes(term)) ||
          (item.subtags && item.subtags.some(subtag => subtag.toLowerCase().includes(term)))
        );
      })
      .map(item => {
        // Create a preview snippet
        let snippet = item.content;
        
        // Find the first matching term occurrence for snippet
        const firstTerm = terms[0];
        const index = item.content.toLowerCase().indexOf(firstTerm);
        
        if (index > 30) {
          snippet = '...' + item.content.substring(index - 20);
        }
        
        if (snippet.length > 150) {
          snippet = snippet.substring(0, 150) + '...';
        }
        
        return {
          ...item,
          snippet
        };
      });
  }
  
  /**
   * Display search results in the UI
   */
  displayResults(results, container) {
    container.innerHTML = '';
    container.classList.remove('hidden');
    
    if (results.length === 0) {
      container.innerHTML = '<div class="search-no-results">No results found</div>';
      return;
    }
    
    results.forEach(result => {
      const resultItem = document.createElement('div');
      resultItem.className = 'search-result-item';
      
      resultItem.innerHTML = `
        <a href="${result.url}" class="search-result-title">${result.title}</a>
        <div class="search-result-meta">
          <span class="date">${result.date}</span>
          <span class="tags">${result.tags.join(', ')}</span>
        </div>
        <div class="search-result-snippet">${result.snippet}</div>
      `;
      
      container.appendChild(resultItem);
    });
  }
}

// Initialize search engine
document.addEventListener('DOMContentLoaded', function() {
  const searchEngine = new SearchEngine();
  searchEngine.init();
});