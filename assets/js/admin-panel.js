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
      draftsSection.innerHTML = '<h4>Drafts</h4><p class="section-description">Unpublished content that you\'re working on.</p>';
      contentManagement.appendChild(draftsSection);
    }
    
    if (!document.getElementById('posts-section')) {
      const postsSection = document.createElement('div');
      postsSection.id = 'posts-section';
      postsSection.innerHTML = '<h4>Published Posts</h4><p class="section-description">Your published content that\'s visible to everyone.</p>';
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
      logoutBtn.className = 'secondary-btn';
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
      renderPosts();
      
      // Clear password field
      document.getElementById('admin-password').value = '';
    } else {
      console.log('Login failed');
      alert('Incorrect password. Access denied.');
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
      renderPosts();
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
    
    // Scroll to form
    postForm.scrollIntoView({ behavior: 'smooth' });
    
    // Initialize Bear Blog style editor
    initBearBlogEditor();
  }
  
  /**
   * Initialize Bear Blog style editor
   */
  function initBearBlogEditor() {
    const actionBar = document.querySelector('.editor-action-bar');
    const editorContainer = document.querySelector('.editor-container');
    
    if (!actionBar || !editorContainer) return;
    
    // Apply initial editor height
    const headerHeight = actionBar.offsetHeight;
    document.documentElement.style.setProperty('--editor-offset', headerHeight + 'px');
    
    // Make sure the action bar stays sticky at the top
    window.addEventListener('scroll', function() {
      const rect = editorContainer.getBoundingClientRect();
      if (rect.top < 0) {
        actionBar.classList.add('sticky');
      } else {
        actionBar.classList.remove('sticky');
      }
    });
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
  
  /**
   * Render drafts in the admin panel
   */
  function renderDrafts() {
    const draftsSection = document.getElementById('drafts-section');
    if (!draftsSection) return;
    
    // Clear existing content
    draftsSection.innerHTML = '<h4>Drafts</h4><p class="section-description">Unpublished content that you\'re working on.</p>';
    
    // Get drafts from storage
    const drafts = dataStorage.getDrafts();
    
    if (drafts.length === 0) {
      draftsSection.innerHTML += '<p>No drafts yet.</p>';
    } else {
      // Create a table for drafts
      const draftsTable = document.createElement('table');
      draftsTable.className = 'admin-table';
      draftsTable.innerHTML = `
        <thead>
          <tr>
            <th>Title</th>
            <th>Date</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${drafts.map(draft => `
            <tr data-id="${draft.id}">
              <td>${draft.title || 'Untitled'}</td>
              <td>${draft.date || 'No date'}</td>
              <td>
                <button class="edit-btn edit-draft-btn">Edit</button>
                <button class="publish-btn publish-draft-btn">Publish</button>
                <button class="delete-btn delete-draft-btn">Delete</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      `;
      
      draftsSection.appendChild(draftsTable);
      
      // Add event listeners for draft actions
      const editDraftBtns = draftsTable.querySelectorAll('.edit-draft-btn');
      const publishDraftBtns = draftsTable.querySelectorAll('.publish-draft-btn');
      const deleteDraftBtns = draftsTable.querySelectorAll('.delete-draft-btn');
      
      editDraftBtns.forEach(btn => {
        btn.addEventListener('click', function() {
          const draftId = this.closest('tr').dataset.id;
          editDraft(draftId);
        });
      });
      
      publishDraftBtns.forEach(btn => {
        btn.addEventListener('click', function() {
          const draftId = this.closest('tr').dataset.id;
          publishDraft(draftId);
        });
      });
      
      deleteDraftBtns.forEach(btn => {
        btn.addEventListener('click', function() {
          const draftId = this.closest('tr').dataset.id;
          deleteDraft(draftId);
        });
      });
    }
  }
  
  /**
   * Render posts in the admin panel
   */
  function renderPosts() {
    const postsSection = document.getElementById('posts-section');
    if (!postsSection) return;
    
    // Clear existing content
    postsSection.innerHTML = '<h4>Published Posts</h4><p class="section-description">Your published content that\'s visible to everyone.</p>';
    
    // Get posts from storage
    const posts = dataStorage.getPosts();
    
    if (posts.length === 0) {
      postsSection.innerHTML += '<p>No published posts yet.</p>';
    } else {
      // Create a table for posts
      const postsTable = document.createElement('table');
      postsTable.className = 'admin-table';
      postsTable.innerHTML = `
        <thead>
          <tr>
            <th>Title</th>
            <th>Date</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${posts.map(post => `
            <tr data-id="${post.id}">
              <td>${post.title || 'Untitled'}</td>
              <td>${post.date || 'No date'}</td>
              <td>
                <button class="edit-btn edit-post-btn">Edit</button>
                <button class="delete-btn delete-post-btn">Delete</button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      `;
      
      postsSection.appendChild(postsTable);
      
      // Add event listeners for post actions
      const editPostBtns = postsTable.querySelectorAll('.edit-post-btn');
      const deletePostBtns = postsTable.querySelectorAll('.delete-post-btn');
      
      editPostBtns.forEach(btn => {
        btn.addEventListener('click', function() {
          const postId = this.closest('tr').dataset.id;
          editPost(postId);
        });
      });
      
      deletePostBtns.forEach(btn => {
        btn.addEventListener('click', function() {
          const postId = this.closest('tr').dataset.id;
          deletePost(postId);
        });
      });
    }
  }
  
  /**
   * Edit a draft
   */
  function editDraft(draftId) {
    const draft = dataStorage.getDrafts().find(d => d.id === draftId);
    if (!draft) return;
    
    const postForm = document.getElementById('post-form');
    const postEditor = document.getElementById('post-editor');
    
    // Fill form with draft data
    document.getElementById('post-title').value = draft.title || '';
    document.getElementById('post-date').value = draft.date || '';
    document.getElementById('post-tags').value = draft.tags.join(', ');
    document.getElementById('post-subtags').value = draft.subtags.join(', ');
    document.getElementById('post-content').value = draft.content || '';
    document.getElementById('post-draft').checked = true;
    
    // Set edit mode
    postEditor.dataset.editId = draftId;
    postEditor.dataset.editType = 'edit-draft';
    
    // Show form
    postForm.classList.remove('hidden');
    
    // Scroll to form
    postForm.scrollIntoView({ behavior: 'smooth' });
    
    // Initialize Bear Blog style editor
    initBearBlogEditor();
    
    // Trigger content preview update
    const contentInput = document.getElementById('post-content');
    if (contentInput && contentInput.value) {
      const event = new Event('input');
      contentInput.dispatchEvent(event);
    }
  }
  
  /**
   * Edit a post
   */
  function editPost(postId) {
    const post = dataStorage.getPosts().find(p => p.id === postId);
    if (!post) return;
    
    const postForm = document.getElementById('post-form');
    const postEditor = document.getElementById('post-editor');
    
    // Fill form with post data
    document.getElementById('post-title').value = post.title || '';
    document.getElementById('post-date').value = post.date || '';
    document.getElementById('post-tags').value = post.tags.join(', ');
    document.getElementById('post-subtags').value = post.subtags.join(', ');
    document.getElementById('post-content').value = post.content || '';
    document.getElementById('post-draft').checked = false;
    
    // Set edit mode
    postEditor.dataset.editId = postId;
    postEditor.dataset.editType = 'edit-post';
    
    // Show form
    postForm.classList.remove('hidden');
    
    // Scroll to form
    postForm.scrollIntoView({ behavior: 'smooth' });
    
    // Initialize Bear Blog style editor
    initBearBlogEditor();
    
    // Trigger content preview update
    const contentInput = document.getElementById('post-content');
    if (contentInput && contentInput.value) {
      const event = new Event('input');
      contentInput.dispatchEvent(event);
    }
  }
  
  /**
   * Publish a draft
   */
  function publishDraft(draftId) {
    if (confirm('Are you sure you want to publish this draft?')) {
      dataStorage.publishDraft(draftId);
      renderDrafts();
      renderPosts();
    }
  }
  
  /**
   * Delete a draft
   */
  function deleteDraft(draftId) {
    if (confirm('Are you sure you want to delete this draft?')) {
      dataStorage.deleteDraft(draftId);
      renderDrafts();
    }
  }
  
  /**
   * Delete a post
   */
  function deletePost(postId) {
    if (confirm('Are you sure you want to delete this post?')) {
      dataStorage.deletePost(postId);
      renderPosts();
    }
  }
});