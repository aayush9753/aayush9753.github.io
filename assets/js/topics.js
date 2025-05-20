document.addEventListener('DOMContentLoaded', function() {
  const topicSearch = document.getElementById('topic-search');
  const topicCards = document.querySelectorAll('.topic-card');
  
  if (topicSearch) {
    topicSearch.addEventListener('input', function() {
      const searchTerm = this.value.trim().toLowerCase();
      filterTopics(searchTerm);
    });
  }
  
  function filterTopics(searchTerm) {
    let visibleCount = 0;
    
    topicCards.forEach(card => {
      const title = card.getAttribute('data-title');
      
      if (title.includes(searchTerm)) {
        card.style.display = 'block';
        visibleCount++;
      } else {
        card.style.display = 'none';
      }
    });
    
    // Check if no results
    const topicsGrid = document.querySelector('.topics-grid');
    
    if (topicsGrid) {
      let noResultsMsg = topicsGrid.querySelector('.no-results-message');
      
      if (visibleCount === 0) {
        if (!noResultsMsg) {
          noResultsMsg = document.createElement('p');
          noResultsMsg.className = 'no-results-message';
          topicsGrid.appendChild(noResultsMsg);
        }
        noResultsMsg.textContent = `No topics match "${searchTerm}".`;
      } else if (noResultsMsg) {
        noResultsMsg.remove();
      }
    }
  }
});