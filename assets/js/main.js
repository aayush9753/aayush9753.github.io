/**
 * MyCave - Main JavaScript
 * Handles view toggling, content organization, and admin functionality
 */

document.addEventListener('DOMContentLoaded', function() {
  // View Toggle Functionality
  setupViewToggles();
  
  // Render posts from storage
  renderPosts();
  
  // Setup admin panel
  setupAdminPanel();
});

/**
 * Set up the view toggle buttons to switch between different content views
 */
function setupViewToggles() {
  const dateViewBtn = document.getElementById('date-view-btn');
  const tagViewBtn = document.getElementById('tag-view-btn');
  const subtagViewBtn = document.getElementById('subtag-view-btn');
  const adminViewBtn = document.getElementById('admin-view-btn');
  
  const dateView = document.getElementById('date-view');
  const tagView = document.getElementById('tag-view');
  const subtagView = document.getElementById('subtag-view');
  const adminPanel = document.getElementById('admin-panel');
  
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
  
  // Admin View Toggle
  adminViewBtn.addEventListener('click', function() {
    setActiveView(adminViewBtn, adminPanel);
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
 * Render posts from data storage to the page
 */
function renderPosts() {
  // Get posts from storage
  const posts = dataStorage.getPosts();
  
  // Sort posts by date (newest first)
  posts.sort((a, b) => new Date(b.date) - new Date(a.date));
  
  // Render in date view
  renderDateView(posts);
  
  // Render in tag view
  renderTagView(posts);
  
  // Render in subtag view
  renderSubtagView(posts);
}

/**
 * Render posts in the date view
 */
function renderDateView(posts) {
  const postList = document.querySelector('#date-view .post-list');
  
  if (!postList) return;
  
  // Clear existing content
  postList.innerHTML = '';
  
  // Add each post
  posts.forEach(post => {
    const li = document.createElement('li');
    
    li.innerHTML = `
      <span class="date">${post.date}</span>
      <a href="${post.url}" class="post-title">${post.title}</a>
      <div class="tags">
        ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
        ${post.subtags ? post.subtags.map(subtag => `<span class="subtag">${subtag}</span>`).join('') : ''}
      </div>
    `;
    
    postList.appendChild(li);
  });
}

/**
 * Render posts in the tag view
 */
function renderTagView(posts) {
  const tagView = document.getElementById('tag-view');
  
  if (!tagView) return;
  
  // Get all unique tags
  const allTags = new Set();
  posts.forEach(post => post.tags.forEach(tag => allTags.add(tag)));
  
  // Update tag list
  const tagList = tagView.querySelector('.tag-list');
  tagList.innerHTML = '';
  
  allTags.forEach(tag => {
    const tagLink = document.createElement('a');
    tagLink.href = `#${tag}`;
    tagLink.className = 'tag';
    tagLink.textContent = tag;
    tagList.appendChild(tagLink);
  });
  
  // Clear existing tag sections
  const existingTagSections = tagView.querySelectorAll('.tag-section');
  existingTagSections.forEach(section => section.remove());
  
  // Create a section for each tag
  allTags.forEach(tag => {
    const tagPosts = posts.filter(post => post.tags.includes(tag));
    
    const section = document.createElement('div');
    section.id = tag;
    section.className = 'tag-section';
    
    section.innerHTML = `
      <h4>${tag}</h4>
      <ul class="post-list">
        ${tagPosts.map(post => `
          <li>
            <span class="date">${post.date}</span>
            <a href="${post.url}" class="post-title">${post.title}</a>
            <div class="tags">
              ${post.subtags ? post.subtags.map(subtag => `<span class="subtag">${subtag}</span>`).join('') : ''}
            </div>
          </li>
        `).join('')}
      </ul>
    `;
    
    tagView.appendChild(section);
  });
  
  // Setup tag filtering
  setupTagFiltering();
}

/**
 * Render posts in the subtag view
 */
function renderSubtagView(posts) {
  const subtagView = document.getElementById('subtag-view');
  
  if (!subtagView) return;
  
  // Get all unique subtags
  const allSubtags = new Set();
  posts.forEach(post => {
    if (post.subtags && Array.isArray(post.subtags)) {
      post.subtags.forEach(subtag => allSubtags.add(subtag));
    }
  });
  
  // Update subtag list
  const subtagList = subtagView.querySelector('.subtag-list');
  subtagList.innerHTML = '';
  
  allSubtags.forEach(subtag => {
    const subtagLink = document.createElement('a');
    subtagLink.href = `#${subtag}`;
    subtagLink.className = 'subtag';
    subtagLink.textContent = subtag;
    subtagList.appendChild(subtagLink);
  });
  
  // Clear existing subtag sections
  const existingSubtagSections = subtagView.querySelectorAll('.subtag-section');
  existingSubtagSections.forEach(section => section.remove());
  
  // Create a section for each subtag
  allSubtags.forEach(subtag => {
    const subtagPosts = posts.filter(post => post.subtags && post.subtags.includes(subtag));
    
    const section = document.createElement('div');
    section.id = subtag;
    section.className = 'subtag-section';
    
    section.innerHTML = `
      <h4>${subtag}</h4>
      <ul class="post-list">
        ${subtagPosts.map(post => `
          <li>
            <span class="date">${post.date}</span>
            <a href="${post.url}" class="post-title">${post.title}</a>
            <div class="tags">
              ${post.tags ? post.tags.map(tag => `<span class="tag">${tag}</span>`).join('') : ''}
            </div>
          </li>
        `).join('')}
      </ul>
    `;
    
    subtagView.appendChild(section);
  });
  
  // Setup subtag filtering
  setupSubtagFiltering();
}

/**
 * Setup the admin panel functionality
 */
function setupAdminPanel() {
  const adminPanel = document.getElementById('admin-panel');
  const loginSection = document.getElementById('login-section');
  const loginBtn = document.getElementById('login-btn');
  const contentManagement = document.getElementById('content-management');
  const newPostBtn = document.getElementById('new-post-btn');
  const postForm = document.getElementById('post-form');
  const cancelPostBtn = document.getElementById('cancel-post');
  
  // Check if already logged in
  console.log('Checking admin status:', dataStorage.checkAdminStatus());
  if (dataStorage.checkAdminStatus()) {
    console.log('Admin already logged in, showing management panel');
    loginSection.style.display = 'none';
    contentManagement.classList.remove('hidden');
    
    // Load and display drafts and posts
    renderDrafts();
  } else {
    console.log('Admin not logged in, showing login screen');
    loginSection.style.display = 'block';
    contentManagement.classList.add('hidden');
  }
  
  // Handle login
  loginBtn.addEventListener('click', function() {
    const password = document.getElementById('admin-password').value;
    
    if (password) {
      const success = dataStorage.adminLogin(password);
      
      if (success) {
        // Hide login section and show content management
        loginSection.style.display = 'none';
        contentManagement.classList.remove('hidden');
        
        // Load and display drafts and posts
        renderDrafts();
        
        console.log('Admin login successful');
      } else {
        alert('Incorrect password. Access denied.');
      }
    } else {
      alert('Please enter the admin password.');
    }
  });
  
  // Also handle login on Enter key press in password field
  document.getElementById('admin-password').addEventListener('keyup', function(event) {
    if (event.key === 'Enter') {
      loginBtn.click();
    }
  });
  
  // Add a logout button if it doesn't exist
  if (!document.getElementById('logout-btn')) {
    const logoutBtn = document.createElement('button');
    logoutBtn.id = 'logout-btn';
    logoutBtn.textContent = 'Logout';
    logoutBtn.style.marginLeft = '10px';
    logoutBtn.style.marginBottom = '15px';
    logoutBtn.addEventListener('click', function() {
      dataStorage.adminLogout();
      loginSection.style.display = 'block';
      contentManagement.classList.add('hidden');
      console.log('Admin logged out');
    });
    
    contentManagement.insertBefore(logoutBtn, contentManagement.firstChild);
  }
  
  // Ensure drafts section exists
  if (!document.getElementById('drafts-section')) {
    const draftsSection = document.createElement('div');
    draftsSection.id = 'drafts-section';
    draftsSection.innerHTML = '<h4>Your Drafts</h4>';
    contentManagement.appendChild(draftsSection);
  }
  
  // Ensure posts section exists
  if (!document.getElementById('posts-section')) {
    const postsSection = document.createElement('div');
    postsSection.id = 'posts-section';
    postsSection.innerHTML = '<h4>Manage Published Posts</h4>';
    contentManagement.appendChild(postsSection);
  }
  
  // Show new post form
  newPostBtn.addEventListener('click', function() {
    // Reset the form and clear current edit ID
    document.getElementById('post-editor').reset();
    document.getElementById('post-editor').dataset.editId = '';
    document.getElementById('post-editor').dataset.editType = 'new';
    
    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('post-date').value = today;
    
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
    const editId = this.dataset.editId;
    const editType = this.dataset.editType;
    
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
    this.reset();
    postForm.classList.add('hidden');
  });
}

/**
 * Render drafts and posts in the admin panel
 */
function renderDrafts() {
  const contentManagement = document.getElementById('content-management');
  const draftsSection = document.getElementById('drafts-section');
  const postsSection = document.getElementById('posts-section');
  
  if (!draftsSection || !postsSection) return;
  
  // Clear existing content
  draftsSection.innerHTML = '<h4>Your Drafts</h4>';
  postsSection.innerHTML = '<h4>Manage Published Posts</h4>';
  
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
              <button class="edit-draft-btn">Edit</button>
              <button class="publish-draft-btn">Publish</button>
              <button class="delete-draft-btn">Delete</button>
            </td>
          </tr>
        `).join('')}
      </tbody>
    `;
    
    draftsSection.appendChild(draftsTable);
  }
  
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
              <button class="edit-post-btn">Edit</button>
              <button class="delete-post-btn">Delete</button>
            </td>
          </tr>
        `).join('')}
      </tbody>
    `;
    
    postsSection.appendChild(postsTable);
  }
  
  // Add some CSS for tables if it doesn't exist
  if (!document.getElementById('admin-table-css')) {
    const style = document.createElement('style');
    style.id = 'admin-table-css';
    style.textContent = `
      .admin-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: var(--spacing-unit);
      }
      
      .admin-table th,
      .admin-table td {
        padding: calc(var(--spacing-unit) * 0.5);
        text-align: left;
        border-bottom: 1px solid var(--border-color);
      }
      
      .admin-table th {
        font-weight: 500;
        background-color: var(--subtle-bg);
      }
      
      .admin-table button {
        margin-right: 5px;
        padding: 3px 8px;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        font-size: calc(var(--base-size) * 0.8);
      }
      
      .admin-table .edit-draft-btn,
      .admin-table .edit-post-btn {
        background-color: #4CAF50;
        color: white;
      }
      
      .admin-table .publish-draft-btn {
        background-color: var(--accent-color);
        color: white;
      }
      
      .admin-table .delete-draft-btn,
      .admin-table .delete-post-btn {
        background-color: #f44336;
        color: white;
      }
    `;
    document.head.appendChild(style);
  }
  
  // Add event listeners for draft actions
  const editDraftBtns = document.querySelectorAll('.edit-draft-btn');
  const publishDraftBtns = document.querySelectorAll('.publish-draft-btn');
  const deleteDraftBtns = document.querySelectorAll('.delete-draft-btn');
  
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
  
  // Add event listeners for post actions
  const editPostBtns = document.querySelectorAll('.edit-post-btn');
  const deletePostBtns = document.querySelectorAll('.delete-post-btn');
  
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

/**
 * Edit a draft
 */
function editDraft(draftId) {
  const draft = dataStorage.getDrafts().find(d => d.id === draftId);
  
  if (draft) {
    // Populate the form with draft data
    const form = document.getElementById('post-editor');
    const postForm = document.getElementById('post-form');
    
    form.dataset.editId = draftId;
    form.dataset.editType = 'edit-draft';
    
    document.getElementById('post-title').value = draft.title || '';
    document.getElementById('post-date').value = draft.date || '';
    document.getElementById('post-tags').value = draft.tags ? draft.tags.join(', ') : '';
    document.getElementById('post-subtags').value = draft.subtags ? draft.subtags.join(', ') : '';
    document.getElementById('post-content').value = draft.content || '';
    document.getElementById('post-draft').checked = true;
    
    // Show the form
    postForm.classList.remove('hidden');
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
 * Edit a post
 */
function editPost(postId) {
  const post = dataStorage.getPosts().find(p => p.id === postId);
  
  if (post) {
    // Populate the form with post data
    const form = document.getElementById('post-editor');
    const postForm = document.getElementById('post-form');
    
    form.dataset.editId = postId;
    form.dataset.editType = 'edit-post';
    
    document.getElementById('post-title').value = post.title || '';
    document.getElementById('post-date').value = post.date || '';
    document.getElementById('post-tags').value = post.tags ? post.tags.join(', ') : '';
    document.getElementById('post-subtags').value = post.subtags ? post.subtags.join(', ') : '';
    document.getElementById('post-content').value = post.content || '';
    document.getElementById('post-draft').checked = false;
    
    // Show the form
    postForm.classList.remove('hidden');
  }
}

/**
 * Delete a post
 */
function deletePost(postId) {
  if (confirm('Are you sure you want to delete this post?')) {
    dataStorage.deletePost(postId);
    renderDrafts();
    renderPosts();
  }
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