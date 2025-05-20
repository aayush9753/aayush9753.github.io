document.addEventListener('DOMContentLoaded', function() {
  // Set up citation tabs
  const citationTabs = document.querySelectorAll('.citation-tab');
  const citationFormats = document.querySelectorAll('.citation-format');
  
  citationTabs.forEach(tab => {
    tab.addEventListener('click', function() {
      // Update active tab
      citationTabs.forEach(t => t.classList.remove('active'));
      this.classList.add('active');
      
      // Show corresponding citation format
      const format = this.getAttribute('data-format');
      citationFormats.forEach(f => f.classList.remove('active'));
      document.getElementById(`${format}-citation`).classList.add('active');
      
      // Update copy button target
      const copyBtn = document.querySelector('.copy-citation-btn');
      copyBtn.setAttribute('data-clipboard-target', `#${format}-citation`);
    });
  });
  
  // Set up clipboard copying
  const clipboard = new ClipboardJS('.copy-citation-btn');
  
  clipboard.on('success', function(e) {
    const copyBtn = e.trigger;
    const originalText = copyBtn.textContent;
    
    copyBtn.textContent = 'Copied!';
    setTimeout(() => {
      copyBtn.textContent = originalText;
    }, 2000);
    
    e.clearSelection();
  });
  
  clipboard.on('error', function(e) {
    const copyBtn = e.trigger;
    copyBtn.textContent = 'Failed to copy';
    setTimeout(() => {
      copyBtn.textContent = 'Copy Citation';
    }, 2000);
  });

  // Handle progress bar animation
  const progressBars = document.querySelectorAll('.progress-bar');
  progressBars.forEach(bar => {
    const fill = bar.querySelector('.progress-fill');
    if (fill) {
      // Add a slight delay for visual effect
      setTimeout(() => {
        fill.style.transition = 'width 1s ease-in-out';
      }, 100);
    }
  });
});