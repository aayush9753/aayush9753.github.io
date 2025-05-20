/**
 * Search Functionality for My Second Brain
 * Provides client-side search across posts
 */

class SearchEngine {
  constructor() {
    this.searchIndex = [];
    this.searchable = false;
  }
  
  /**
   * Initialize search with mock data
   * In a real implementation, this would load from a JSON index
   */
  init() {
    // Mock search index for demonstration
    this.searchIndex = [
      {
        id: 'post-1',
        title: 'Notes on building a second brain',
        date: '2023-07-15',
        tags: ['productivity', 'knowledge-management'],
        subtags: ['note-taking', 'zettelkasten'],
        content: 'The concept of a "second brain" has gained significant traction in productivity circles. At its core, it\'s about creating an external system to store, organize, and retrieve the information we consume, helping us think more clearly and create more effectively.',
        url: 'post-template.html'
      },
      {
        id: 'post-2',
        title: 'Reflections on minimalism',
        date: '2023-06-30',
        tags: ['lifestyle', 'philosophy'],
        subtags: ['digital-minimalism', 'simplicity'],
        content: 'Minimalism is more than an aesthetic choice; it's a mindset that prioritizes value over volume. By deliberately choosing what to include in our lives, we create space for what truly matters.',
        url: '#'
      },
      {
        id: 'post-3',
        title: 'Understanding plain text productivity',
        date: '2023-06-10',
        tags: ['productivity', 'tools'],
        subtags: ['markdown', 'text-files'],
        content: 'Plain text systems offer surprising advantages for productivity: they're portable, future-proof, and distraction-free. By embracing the constraints of simple text files, we can focus on what matters: our thoughts and ideas.',
        url: '#'
      },
      {
        id: 'post-4',
        title: 'Digital gardens vs traditional blogs',
        date: '2023-05-22',
        tags: ['writing', 'web'],
        subtags: ['digital-gardens', 'publishing'],
        content: 'Unlike chronological blogs, digital gardens are non-linear, continuously evolving collections of thoughts and notes. They emphasize connection and growth over temporal organization.',
        url: '#'
      },
      {
        id: 'post-5',
        title: 'Getting started with personal knowledge management',
        date: '2023-05-01',
        tags: ['productivity', 'knowledge-management'],
        subtags: ['note-taking', 'organization'],
        content: 'Personal knowledge management is about systematically capturing, organizing, and sharing what you know and learn. It begins with reliable capture methods and requires consistent maintenance.',
        url: '#'
      }
    ];
    
    this.searchable = true;
    
    // Set up search UI if it exists
    this.setupSearchUI();
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
    
    const terms = query.toLowerCase().split(' ');
    return this.searchIndex
      .filter(item => {
        // Check if all terms are found somewhere in the item
        return terms.every(term => 
          item.title.toLowerCase().includes(term) ||
          item.content.toLowerCase().includes(term) ||
          item.tags.some(tag => tag.toLowerCase().includes(term)) ||
          item.subtags.some(subtag => subtag.toLowerCase().includes(term))
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