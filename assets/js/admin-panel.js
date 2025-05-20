/**
 * Admin Panel Module for MyCave
 * Handles admin authentication and content management
 */

document.addEventListener('DOMContentLoaded', function() {
  if (!dataStorage) {
    console.error('Cannot initialize admin panel: dataStorage not found');
    return;
  }

  console.log('Initializing admin panel...');
  
  // Get the required DOM elements
  const adminPanel = document.getElementById('admin-panel');
  if (!adminPanel) {
    console.error('Cannot find admin panel in the DOM');
    return;
  }
  
  // Reset admin status (for debugging)
  // Uncomment the line below to reset admin status
  // dataStorage.resetAdminStatus();
  
  setupAdminPanel();
  
  /**
   * Setup the admin panel functionality
   */
  function setupAdminPanel() {
    const loginSection = document.getElementById('login-section');
    const adminPassword = document.getElementById('admin-password');
    const loginBtn = document.getElementById('login-btn');
    const contentManagement = document.getElementById('content-management');
    const newPostBtn = document.getElementById('new-post-btn');
    const postForm = document.getElementById('post-form');
    const cancelPostBtn = document.getElementById('cancel-post');
    
    // Ensure DOM elements exist
    if (!loginSection || !adminPassword || !loginBtn || !contentManagement) {
      console.error('Missing required DOM elements for admin panel');
      return;
    }
    
    // Create DOM elements if they don't exist
    if (!document.getElementById('drafts-section')) {
      const draftsSection = document.createElement('div');
      draftsSection.id = 'drafts-section';
      draftsSection.innerHTML = '<h4>Your Drafts</h4>';
      contentManagement.appendChild(draftsSection);
    }
    
    if (!document.getElementById('posts-section')) {
      const postsSection = document.createElement('div');
      postsSection.id = 'posts-section';
      postsSection.innerHTML = '<h4>Manage Published Posts</h4>';
      contentManagement.appendChild(postsSection);
    }
    
    // Check if already logged in
    updateAdminUI();
    
    // Add login event handler
    loginBtn.addEventListener('click', handleLogin);
    
    // Add Enter key press handler for admin password
    adminPassword.addEventListener('keyup', function(event) {
      if (event.key === 'Enter') {
        handleLogin();
      }
    });
    
    // Add logout button if it doesn't exist
    if (!document.getElementById('logout-btn')) {
      const logoutBtn = document.createElement('button');
      logoutBtn.id = 'logout-btn';
      logoutBtn.textContent = 'Logout';
      logoutBtn.style.marginLeft = '10px';
      logoutBtn.style.marginBottom = '15px';
      logoutBtn.addEventListener('click', handleLogout);
      
      contentManagement.insertBefore(logoutBtn, contentManagement.firstChild);
    }
    
    // Add new post form event handlers
    if (newPostBtn && postForm && cancelPostBtn) {
      // Show new post form
      newPostBtn.addEventListener('click', showNewPostForm);
      
      // Cancel post creation/editing
      cancelPostBtn.addEventListener('click', hidePostForm);
      
      // Handle post submission
      const postEditor = document.getElementById('post-editor');
      if (postEditor) {
        postEditor.addEventListener('submit', handlePostSubmission);
      } else {
        console.error('Post editor form not found');
      }
    }
    
    console.log('Admin panel setup complete');
  }
  
  /**
   * Handle admin login
   */
  function handleLogin() {
    const password = document.getElementById('admin-password').value;
    const loginSection = document.getElementById('login-section');
    const contentManagement = document.getElementById('content-management');
    
    if (!password) {
      alert('Please enter the admin password.');
      return;
    }
    
    console.log('Attempting login with password:', password);
    const success = dataStorage.adminLogin(password);
    
    if (success) {
      console.log('Login successful');
      
      // Update UI
      loginSection.style.display = 'none';
      contentManagement.classList.remove('hidden');
      
      // Render drafts and posts
      renderDrafts();
      
      // Clear password field
      document.getElementById('admin-password').value = '';
    } else {
      console.log('Login failed');
      alert('Incorrect password. Access denied. The correct password is "admin123".');
    }
  }
  
  /**
   * Handle admin logout
   */
  function handleLogout() {
    const loginSection = document.getElementById('login-section');
    const contentManagement = document.getElementById('content-management');
    
    dataStorage.adminLogout();
    
    loginSection.style.display = 'block';
    contentManagement.classList.add('hidden');
    
    console.log('Admin logged out');
  }
  
  /**
   * Update admin UI based on login status
   */
  function updateAdminUI() {
    const loginSection = document.getElementById('login-section');
    const contentManagement = document.getElementById('content-management');
    
    const isAdmin = dataStorage.checkAdminStatus();
    console.log('Updating admin UI. isAdmin =', isAdmin);
    
    if (isAdmin) {
      loginSection.style.display = 'none';
      contentManagement.classList.remove('hidden');
      renderDrafts();
    } else {
      loginSection.style.display = 'block';
      contentManagement.classList.add('hidden');
    }
  }
  
  /**
   * Show new post form
   */
  function showNewPostForm() {
    const postForm = document.getElementById('post-form');
    const postEditor = document.getElementById('post-editor');
    
    // Reset the form and clear current edit ID
    postEditor.reset();
    postEditor.dataset.editId = '';
    postEditor.dataset.editType = 'new';
    
    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('post-date').value = today;
    
    postForm.classList.remove('hidden');
  }
  
  /**
   * Hide post form
   */
  function hidePostForm() {
    const postForm = document.getElementById('post-form');
    const postEditor = document.getElementById('post-editor');
    
    postForm.classList.add('hidden');
    postEditor.reset();
  }
  
  /**
   * Handle post submission
   */
  function handlePostSubmission(e) {
    e.preventDefault();
    
    // Get the form
    const form = this;
    
    // Collect form data
    const postData = {
      title: document.getElementById('post-title').value,
      date: document.getElementById('post-date').value,
      tags: document.getElementById('post-tags').value.split(',')
        .map(tag => tag.trim())
        .filter(tag => tag), // Remove empty strings
      subtags: document.getElementById('post-subtags').value.split(',')
        .map(subtag => subtag.trim())
        .filter(subtag => subtag), // Remove empty strings
      content: document.getElementById('post-content').value,
      isDraft: document.getElementById('post-draft').checked
    };
    
    // Ensure tags and subtags are valid arrays
    if (!postData.tags || !Array.isArray(postData.tags)) {
      postData.tags = [];
    }
    
    if (!postData.subtags || !Array.isArray(postData.subtags)) {
      postData.subtags = [];
    }
    
    // Check if we're editing or creating
    const editId = form.dataset.editId;
    const editType = form.dataset.editType;
    
    if (postData.isDraft) {
      // Save as draft
      if (editType === 'edit-draft' && editId) {
        // Update existing draft
        dataStorage.saveDraft({
          ...postData,
          id: editId
        });
      } else {
        // Create new draft
        dataStorage.saveDraft(postData);
      }
      
      alert('Draft saved successfully!');
      renderDrafts();
    } else {
      // Publish post
      if (editType === 'edit-post' && editId) {
        // Update existing post
        dataStorage.updatePost(editId, postData);
      } else if (editType === 'edit-draft' && editId) {
        // Publish draft as post
        dataStorage.publishDraft(editId);
      } else {
        // Create and publish new post
        dataStorage.addPost(postData);
      }
      
      alert('Post published successfully!');
      renderPosts();
      renderDrafts();
    }
    
    // Reset form and hide it
    form.reset();
    document.getElementById('post-form').classList.add('hidden');
  }
});