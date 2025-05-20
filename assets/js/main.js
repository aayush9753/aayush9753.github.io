/**
 * My Second Brain - Main JavaScript
 * Handles view toggling, content organization, and admin functionality
 */

document.addEventListener('DOMContentLoaded', function() {
  // View Toggle Functionality
  setupViewToggles();
  
  // Admin Panel Functionality (if user is authenticated)
  setupAdminPanel();
});

/**
 * Set up the view toggle buttons to switch between different content views
 */
function setupViewToggles() {
  const dateViewBtn = document.getElementById('date-view-btn');
  const tagViewBtn = document.getElementById('tag-view-btn');
  const subtagViewBtn = document.getElementById('subtag-view-btn');
  
  const dateView = document.getElementById('date-view');
  const tagView = document.getElementById('tag-view');
  const subtagView = document.getElementById('subtag-view');
  
  // Date View Toggle
  dateViewBtn.addEventListener('click', function() {
    setActiveView(dateViewBtn, dateView);
  });
  
  // Tag View Toggle
  tagViewBtn.addEventListener('click', function() {
    setActiveView(tagViewBtn, tagView);
  });
  
  // Subtag View Toggle
  subtagViewBtn.addEventListener('click', function() {
    setActiveView(subtagViewBtn, subtagView);
  });
}

/**
 * Sets the active view and button, hiding all others
 */
function setActiveView(activeButton, activeView) {
  // Update button states
  const buttons = document.querySelectorAll('.view-toggles button');
  buttons.forEach(button => button.classList.remove('active'));
  activeButton.classList.add('active');
  
  // Update view visibility
  const views = document.querySelectorAll('.content-view');
  views.forEach(view => view.classList.remove('active'));
  activeView.classList.add('active');
}

/**
 * Sets up the admin panel functionality if user is authenticated
 */
function setupAdminPanel() {
  const adminPanel = document.getElementById('admin-panel');
  const loginBtn = document.getElementById('login-btn');
  const contentManagement = document.getElementById('content-management');
  const newPostBtn = document.getElementById('new-post-btn');
  const postForm = document.getElementById('post-form');
  const cancelPostBtn = document.getElementById('cancel-post');
  
  // Show admin panel with login button
  adminPanel.classList.remove('hidden');
  
  // Google Authentication Setup (placeholder for OAuth implementation)
  loginBtn.addEventListener('click', function() {
    // In a real implementation, this would trigger the Google OAuth flow
    console.log('Google authentication would be triggered here');
    
    // For demonstration, simulate successful login
    mockSuccessfulLogin();
  });
  
  // Show new post form
  newPostBtn.addEventListener('click', function() {
    postForm.classList.remove('hidden');
  });
  
  // Cancel post creation/editing
  cancelPostBtn.addEventListener('click', function() {
    postForm.classList.add('hidden');
    document.getElementById('post-editor').reset();
  });
  
  // Handle post submission
  document.getElementById('post-editor').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Collect form data
    const postData = {
      title: document.getElementById('post-title').value,
      date: document.getElementById('post-date').value,
      tags: document.getElementById('post-tags').value.split(',').map(tag => tag.trim()),
      subtags: document.getElementById('post-subtags').value.split(',').map(subtag => subtag.trim()),
      content: document.getElementById('post-content').value,
      isDraft: document.getElementById('post-draft').checked
    };
    
    // In a real implementation, this would save the post data
    console.log('Post data would be saved:', postData);
    
    // Reset form and hide it
    this.reset();
    postForm.classList.add('hidden');
    
    // Show success message (in a real implementation)
    alert('Post saved successfully!');
  });
}

/**
 * Mock successful login for demonstration purposes
 */
function mockSuccessfulLogin() {
  document.getElementById('login-section').style.display = 'none';
  document.getElementById('content-management').classList.remove('hidden');
}

/**
 * Function to handle tag filtering in tag view
 */
function setupTagFiltering() {
  const tagLinks = document.querySelectorAll('.tag-list a.tag');
  
  tagLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href').substring(1);
      const targetSection = document.getElementById(targetId);
      
      // Scroll to the target section
      if (targetSection) {
        targetSection.scrollIntoView({
          behavior: 'smooth'
        });
      }
    });
  });
}

/**
 * Function to handle subtag filtering in subtag view
 */
function setupSubtagFiltering() {
  const subtagLinks = document.querySelectorAll('.subtag-list a.subtag');
  
  subtagLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href').substring(1);
      const targetSection = document.getElementById(targetId);
      
      // Scroll to the target section
      if (targetSection) {
        targetSection.scrollIntoView({
          behavior: 'smooth'
        });
      }
    });
  });
}

// Initialize tag and subtag filtering when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  setupTagFiltering();
  setupSubtagFiltering();
});