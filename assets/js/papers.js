document.addEventListener('DOMContentLoaded', function() {
  const topicFilter = document.getElementById('topic-filter');
  const tagFilters = document.querySelectorAll('.tag-filter');
  const papersSortSelect = document.getElementById('papers-sort');
  const papersContainer = document.querySelector('.papers-list');
  const paperCards = document.querySelectorAll('.paper-card');
  const activeFiltersList = document.getElementById('active-filters-list');
  const clearFiltersBtn = document.getElementById('clear-filters');
  const showMoreTagsBtn = document.getElementById('show-more-tags');
  const moreTagsDropdown = document.querySelector('.more-tags-dropdown');
  
  // Active filters state
  let activeFilters = {
    topic: 'all',
    tags: []
  };
  
  // Initialize with all papers visible
  updateActiveFiltersDisplay();
  
  // Topic filter change
  if (topicFilter) {
    topicFilter.addEventListener('change', function() {
      activeFilters.topic = this.value;
      applyFilters();
      updateActiveFiltersDisplay();
    });
  }
  
  // Tag filters click
  tagFilters.forEach(tagBtn => {
    tagBtn.addEventListener('click', function() {
      const tag = this.getAttribute('data-tag');
      
      if (this.classList.contains('active')) {
        // Remove tag from active filters
        this.classList.remove('active');
        activeFilters.tags = activeFilters.tags.filter(t => t !== tag);
      } else {
        // Add tag to active filters
        this.classList.add('active');
        activeFilters.tags.push(tag);
      }
      
      applyFilters();
      updateActiveFiltersDisplay();
    });
  });
  
  // Sort change
  if (papersSortSelect) {
    papersSortSelect.addEventListener('change', function() {
      sortPapers(this.value);
    });
  }
  
  // Clear all filters
  if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', function() {
      activeFilters.topic = 'all';
      activeFilters.tags = [];
      
      // Reset UI
      if (topicFilter) topicFilter.value = 'all';
      tagFilters.forEach(btn => btn.classList.remove('active'));
      
      applyFilters();
      updateActiveFiltersDisplay();
    });
  }
  
  // Show more tags dropdown
  if (showMoreTagsBtn) {
    showMoreTagsBtn.addEventListener('click', function() {
      moreTagsDropdown.classList.toggle('visible');
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(event) {
      if (!event.target.matches('#show-more-tags') && 
          !event.target.closest('.more-tags-dropdown')) {
        moreTagsDropdown.classList.remove('visible');
      }
    });
  }
  
  function applyFilters() {
    paperCards.forEach(card => {
      let showCard = true;
      
      // Apply topic filter
      if (activeFilters.topic !== 'all') {
        const cardTopics = card.getAttribute('data-topics').split(',');
        if (!cardTopics.includes(activeFilters.topic)) {
          showCard = false;
        }
      }
      
      // Apply tag filters
      if (activeFilters.tags.length > 0) {
        const cardTags = card.getAttribute('data-tags').split(',').map(t => t.trim());
        const hasMatchingTag = activeFilters.tags.some(tag => cardTags.includes(tag));
        if (!hasMatchingTag) {
          showCard = false;
        }
      }
      
      // Show or hide card
      card.style.display = showCard ? 'block' : 'none';
    });
    
    // Check if no results
    const visibleCards = Array.from(paperCards).filter(card => card.style.display !== 'none');
    
    if (visibleCards.length === 0) {
      let noResultsMsg = document.querySelector('.no-results-message');
      
      if (!noResultsMsg) {
        noResultsMsg = document.createElement('p');
        noResultsMsg.className = 'no-results-message';
        papersContainer.appendChild(noResultsMsg);
      }
      
      noResultsMsg.textContent = 'No papers match the selected filters.';
    } else {
      const noResultsMsg = document.querySelector('.no-results-message');
      if (noResultsMsg) {
        noResultsMsg.remove();
      }
    }
  }
  
  function sortPapers(sortType) {
    const cardsArray = Array.from(paperCards);
    
    switch (sortType) {
      case 'date-desc':
        cardsArray.sort((a, b) => {
          const dateA = new Date(a.getAttribute('data-date'));
          const dateB = new Date(b.getAttribute('data-date'));
          return dateB - dateA;
        });
        break;
      case 'date-asc':
        cardsArray.sort((a, b) => {
          const dateA = new Date(a.getAttribute('data-date'));
          const dateB = new Date(b.getAttribute('data-date'));
          return dateA - dateB;
        });
        break;
      case 'title-asc':
        cardsArray.sort((a, b) => {
          return a.getAttribute('data-title').localeCompare(b.getAttribute('data-title'));
        });
        break;
      case 'title-desc':
        cardsArray.sort((a, b) => {
          return b.getAttribute('data-title').localeCompare(a.getAttribute('data-title'));
        });
        break;
    }
    
    // Append sorted cards back to container
    cardsArray.forEach(card => {
      papersContainer.appendChild(card);
    });
  }
  
  function updateActiveFiltersDisplay() {
    if (!activeFiltersList) return;
    
    // Clear current filters
    activeFiltersList.innerHTML = '';
    
    let hasActiveFilters = false;
    
    // Add topic filter if not "all"
    if (activeFilters.topic !== 'all') {
      hasActiveFilters = true;
      const topicLabel = document.createElement('span');
      topicLabel.className = 'active-filter topic-filter';
      
      // Get the topic name from the select option
      const topicName = topicFilter.options[topicFilter.selectedIndex].text;
      
      topicLabel.innerHTML = `Topic: ${topicName} <button class="remove-filter" data-filter-type="topic">×</button>`;
      activeFiltersList.appendChild(topicLabel);
      
      // Remove topic filter when clicking the × button
      topicLabel.querySelector('.remove-filter').addEventListener('click', function() {
        activeFilters.topic = 'all';
        topicFilter.value = 'all';
        applyFilters();
        updateActiveFiltersDisplay();
      });
    }
    
    // Add tag filters
    activeFilters.tags.forEach(tag => {
      hasActiveFilters = true;
      const tagLabel = document.createElement('span');
      tagLabel.className = 'active-filter tag-filter';
      tagLabel.innerHTML = `Tag: ${tag} <button class="remove-filter" data-filter-type="tag" data-tag="${tag}">×</button>`;
      activeFiltersList.appendChild(tagLabel);
      
      // Remove tag filter when clicking the × button
      tagLabel.querySelector('.remove-filter').addEventListener('click', function() {
        const tagToRemove = this.getAttribute('data-tag');
        activeFilters.tags = activeFilters.tags.filter(t => t !== tagToRemove);
        
        // Update UI
        tagFilters.forEach(btn => {
          if (btn.getAttribute('data-tag') === tagToRemove) {
            btn.classList.remove('active');
          }
        });
        
        applyFilters();
        updateActiveFiltersDisplay();
      });
    });
    
    // Show "None" if no active filters
    if (!hasActiveFilters) {
      const noFilters = document.createElement('span');
      noFilters.className = 'no-filters';
      noFilters.textContent = 'None';
      activeFiltersList.appendChild(noFilters);
    }
  }
});