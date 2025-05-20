document.addEventListener('DOMContentLoaded', function() {
  const priorityFilters = document.querySelectorAll('.priority-filter');
  const queueSortSelect = document.getElementById('queue-sort');
  const paperCards = document.querySelectorAll('.paper-card.queue');
  
  // Priority filter click
  priorityFilters.forEach(filter => {
    filter.addEventListener('click', function() {
      // Update active filter UI
      priorityFilters.forEach(f => f.classList.remove('active'));
      this.classList.add('active');
      
      // Apply filter
      const priority = this.getAttribute('data-priority');
      filterPapersByPriority(priority);
    });
  });
  
  // Sort change
  if (queueSortSelect) {
    queueSortSelect.addEventListener('click', function() {
      sortPapers(this.value);
    });
  }
  
  function filterPapersByPriority(priority) {
    paperCards.forEach(card => {
      if (priority === 'all') {
        card.style.display = 'block';
      } else {
        const cardPriority = card.classList.contains(`priority-${priority}`);
        card.style.display = cardPriority ? 'block' : 'none';
      }
    });
    
    // Check if no papers match the filter
    const visibleCards = Array.from(paperCards).filter(card => card.style.display !== 'none');
    
    const queueList = document.querySelector('.papers-list.queue');
    if (queueList) {
      let noResultsMsg = queueList.querySelector('.no-results-message');
      
      if (visibleCards.length === 0) {
        if (!noResultsMsg) {
          noResultsMsg = document.createElement('p');
          noResultsMsg.className = 'no-results-message';
          queueList.appendChild(noResultsMsg);
        }
        noResultsMsg.textContent = `No papers with ${priority} priority in your queue.`;
      } else if (noResultsMsg) {
        noResultsMsg.remove();
      }
    }
  }
  
  function sortPapers(sortType) {
    const cardsArray = Array.from(paperCards);
    const queueList = document.querySelector('.papers-list.queue');
    
    if (!queueList) return;
    
    switch (sortType) {
      case 'priority':
        // Sort by priority: high -> medium -> low
        cardsArray.sort((a, b) => {
          const priorityOrder = { 'high': 1, 'medium': 2, 'low': 3 };
          const priorityA = a.classList.contains('priority-high') ? 'high' : 
                           (a.classList.contains('priority-medium') ? 'medium' : 'low');
          const priorityB = b.classList.contains('priority-high') ? 'high' : 
                           (b.classList.contains('priority-medium') ? 'medium' : 'low');
          
          return priorityOrder[priorityA] - priorityOrder[priorityB];
        });
        break;
      case 'date-added':
        // Sort by date added to queue (newest first)
        cardsArray.sort((a, b) => {
          const dateA = new Date(a.querySelector('.added-date').textContent.replace('Added on ', ''));
          const dateB = new Date(b.querySelector('.added-date').textContent.replace('Added on ', ''));
          return dateB - dateA;
        });
        break;
      case 'title':
        // Sort alphabetically by title
        cardsArray.sort((a, b) => {
          const titleA = a.querySelector('h3 a').textContent;
          const titleB = b.querySelector('h3 a').textContent;
          return titleA.localeCompare(titleB);
        });
        break;
    }
    
    // Append sorted cards back to container
    cardsArray.forEach(card => {
      queueList.appendChild(card);
    });
  }
});