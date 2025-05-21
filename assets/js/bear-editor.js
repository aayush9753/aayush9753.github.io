/**
 * Bear Blog Style Editor for MyCave
 * Handles YAML-style metadata headers and Markdown content
 * Enhanced with image upload, resizing, full screen mode, and dark mode
 */

document.addEventListener('DOMContentLoaded', function() {
  // Elements
  const postForm = document.getElementById('post-form');
  const postEditor = document.getElementById('post-editor');
  const headersTextarea = document.getElementById('post-headers-textarea');
  const contentTextarea = document.getElementById('post-content');
  const previewButton = document.getElementById('preview-post');
  const previewSection = document.getElementById('post-preview');
  const previewContent = document.querySelector('.preview-content');
  const editorActionBar = document.querySelector('.editor-action-bar');
  
  // Hidden fields for data compatibility
  const titleField = document.getElementById('post-title');
  const dateField = document.getElementById('post-date');
  const tagsField = document.getElementById('post-tags');
  const subtagsField = document.getElementById('post-subtags');
  
  // Check if editor elements exist
  if (!postForm || !postEditor || !headersTextarea || !contentTextarea) {
    console.error('Bear editor elements not found in the DOM');
    return;
  }
  
  // Initialize editor
  initEditor();
  
  /**
   * Initialize the Bear Blog style editor
   */
  function initEditor() {
    console.log('Initializing Bear Blog style editor');
    
    // Add sticky header functionality
    window.addEventListener('scroll', function() {
      const formTop = postForm.getBoundingClientRect().top;
      if (formTop < 0) {
        editorActionBar.classList.add('sticky');
      } else {
        editorActionBar.classList.remove('sticky');
      }
    });
    
    // Preview button click handler
    if (previewButton) {
      previewButton.addEventListener('click', function() {
        togglePreview();
      });
    }
    
    // Handle form submission
    postEditor.addEventListener('submit', function(e) {
      // Parse headers to individual fields before submitting
      parseHeadersToFields();
    });
    
    // Override editDraft, editPost, and showNewPostForm to use Bear Blog style editor
    overrideMethods();
    
    // Initialize enhanced features
    initDarkMode();
    initFullScreenMode();
    initImageUpload();
    initKeyboardShortcuts();
    
    // Set the date automatically
    setAutomaticDate();
    
    // Make title field stand out
    highlightTitleField();
    
    // Make tags and subtags optional
    markOptionalFields();
  }
  
  /**
   * Sets up dark mode toggle
   */
  function initDarkMode() {
    // Create dark mode toggle
    const darkModeToggle = document.createElement('button');
    darkModeToggle.id = 'dark-mode-toggle';
    darkModeToggle.className = 'editor-btn';
    darkModeToggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278z"/></svg>';
    darkModeToggle.title = 'Toggle Dark Mode';
    
    // Append to editor action bar
    if (editorActionBar) {
      const actionBarLeft = editorActionBar.querySelector('.action-bar-left') || editorActionBar;
      actionBarLeft.appendChild(darkModeToggle);
    }
    
    // Check for user preference
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedMode = localStorage.getItem('editorDarkMode');
    const isDarkMode = savedMode ? savedMode === 'true' : prefersDarkMode;
    
    // Set initial mode
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
    }
    
    // Add event listener
    darkModeToggle.addEventListener('click', function() {
      document.body.classList.toggle('dark-mode');
      const isDark = document.body.classList.contains('dark-mode');
      localStorage.setItem('editorDarkMode', isDark.toString());
    });
  }
  
  /**
   * Sets up full screen mode
   */
  function initFullScreenMode() {
    // Create fullscreen toggle
    const fullscreenToggle = document.createElement('button');
    fullscreenToggle.id = 'fullscreen-toggle';
    fullscreenToggle.className = 'editor-btn';
    fullscreenToggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M1.5 1a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 0h4a.5.5 0 0 1 0 1h-4zM10 .5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 16 1.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5zM.5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 14.5v-4a.5.5 0 0 1 .5-.5zm15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5z"/></svg>';
    fullscreenToggle.title = 'Toggle Fullscreen Mode';
    
    // Append to editor action bar
    if (editorActionBar) {
      const actionBarLeft = editorActionBar.querySelector('.action-bar-left') || editorActionBar;
      actionBarLeft.appendChild(fullscreenToggle);
    }
    
    // Add event listener
    fullscreenToggle.addEventListener('click', function() {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        postEditor.requestFullscreen();
      }
    });
    
    // Handle fullscreen change
    document.addEventListener('fullscreenchange', function() {
      if (document.fullscreenElement) {
        fullscreenToggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.5 0a.5.5 0 0 1 .5.5v4A1.5 1.5 0 0 1 4.5 6h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5zm5 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 10 4.5v-4a.5.5 0 0 1 .5-.5zM0 10.5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 6 11.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5zm10 1a1.5 1.5 0 0 1 1.5-1.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4z"/></svg>';
      } else {
        fullscreenToggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M1.5 1a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 0h4a.5.5 0 0 1 0 1h-4zM10 .5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 16 1.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5zM.5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 14.5v-4a.5.5 0 0 1 .5-.5zm15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5z"/></svg>';
      }
    });
  }
  
  /**
   * Sets up image upload functionality
   */
  function initImageUpload() {
    // Create image upload button
    const imageUploadBtn = document.createElement('button');
    imageUploadBtn.id = 'image-upload-btn';
    imageUploadBtn.className = 'editor-btn';
    imageUploadBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"/><path d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2h-12zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1h12z"/></svg>';
    imageUploadBtn.title = 'Upload Image';
    
    // Create hidden file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'image-file-input';
    fileInput.className = 'hidden';
    fileInput.accept = 'image/*';
    
    // Append to editor
    if (editorActionBar) {
      const actionBarLeft = editorActionBar.querySelector('.action-bar-left') || editorActionBar;
      actionBarLeft.appendChild(imageUploadBtn);
      actionBarLeft.appendChild(fileInput);
    }
    
    // Add click event to button
    imageUploadBtn.addEventListener('click', function() {
      fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
      const file = this.files[0];
      if (!file) return;
      
      // Create image upload modal
      const modal = createImageUploadModal(file);
      document.body.appendChild(modal);
      
      // Show modal
      setTimeout(() => {
        modal.classList.add('active');
      }, 10);
    });
  }
  
  /**
   * Creates a modal for image upload preview and options
   */
  function createImageUploadModal(file) {
    // Create modal container
    const modal = document.createElement('div');
    modal.className = 'image-upload-modal';
    
    // Read file as data URL
    const reader = new FileReader();
    reader.onload = function(e) {
      // Create modal content with image preview
      modal.innerHTML = `
        <div class="modal-content">
          <h3>Insert Image</h3>
          <div class="image-preview-container">
            <img src="${e.target.result}" id="image-preview" alt="Preview">
          </div>
          <div class="image-options">
            <div class="form-group">
              <label for="image-alt">Alt Text</label>
              <input type="text" id="image-alt" placeholder="Describe the image">
            </div>
            <div class="form-group">
              <label for="image-width">Width (px or %)</label>
              <input type="text" id="image-width" placeholder="e.g., 300px or 100%">
            </div>
            <div class="form-group">
              <label for="image-align">Alignment</label>
              <select id="image-align">
                <option value="none">None</option>
                <option value="left">Left</option>
                <option value="center">Center</option>
                <option value="right">Right</option>
              </select>
            </div>
          </div>
          <div class="modal-actions">
            <button id="insert-image-btn" class="primary-btn">Insert Image</button>
            <button id="cancel-image-btn" class="secondary-btn">Cancel</button>
          </div>
        </div>
      `;
      
      // Add event listeners
      setTimeout(() => {
        // Insert image button
        const insertBtn = document.getElementById('insert-image-btn');
        if (insertBtn) {
          insertBtn.addEventListener('click', function() {
            insertImageToEditor(file, modal);
          });
        }
        
        // Cancel button
        const cancelBtn = document.getElementById('cancel-image-btn');
        if (cancelBtn) {
          cancelBtn.addEventListener('click', function() {
            closeModal(modal);
          });
        }
        
        // Close on click outside
        modal.addEventListener('click', function(e) {
          if (e.target === modal) {
            closeModal(modal);
          }
        });
      }, 100);
    };
    
    reader.readAsDataURL(file);
    return modal;
  }
  
  /**
   * Close modal with animation
   */
  function closeModal(modal) {
    modal.classList.remove('active');
    setTimeout(() => {
      modal.remove();
    }, 300);
  }
  
  /**
   * Insert image to editor with selected options
   */
  function insertImageToEditor(file, modal) {
    // Get option values
    const altText = document.getElementById('image-alt').value || '';
    const width = document.getElementById('image-width').value || '';
    const align = document.getElementById('image-align').value || 'none';
    
    // Create image data URL
    const reader = new FileReader();
    reader.onload = function(e) {
      // Base markdown image
      let markdownImage = `![${altText}](${e.target.result})`;
      
      // Add custom attributes for width and alignment
      if (width || align !== 'none') {
        markdownImage = `<img src="${e.target.result}" alt="${altText}"`;
        if (width) {
          markdownImage += ` width="${width}"`;
        }
        if (align !== 'none') {
          markdownImage += ` align="${align}"`;
        }
        markdownImage += '>';
      }
      
      // Insert at cursor position
      insertAtCursor(contentTextarea, markdownImage);
      
      // Close modal
      closeModal(modal);
      
      // Trigger content change for preview update
      const event = new Event('input');
      contentTextarea.dispatchEvent(event);
    };
    
    reader.readAsDataURL(file);
  }
  
  /**
   * Initialize keyboard shortcuts (similar to Notion)
   */
  function initKeyboardShortcuts() {
    contentTextarea.addEventListener('keydown', function(e) {
      // Check for modifier key combinations
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;
      
      // Cmd/Ctrl + B: Bold
      if (cmdOrCtrl && e.key === 'b') {
        e.preventDefault();
        formatText('bold');
      }
      
      // Cmd/Ctrl + I: Italic
      else if (cmdOrCtrl && e.key === 'i') {
        e.preventDefault();
        formatText('italic');
      }
      
      // Cmd/Ctrl + K: Link
      else if (cmdOrCtrl && e.key === 'k') {
        e.preventDefault();
        formatText('link');
      }
      
      // Cmd/Ctrl + H + 1-6: Headings
      else if (cmdOrCtrl && e.key === 'h') {
        // Wait for the next key
        const currentPos = contentTextarea.selectionStart;
        
        const headingHandler = function(event) {
          // Remove this handler after use
          document.removeEventListener('keydown', headingHandler);
          
          if (event.key >= '1' && event.key <= '6') {
            event.preventDefault();
            const level = parseInt(event.key);
            formatText('heading', level);
          }
        };
        
        document.addEventListener('keydown', headingHandler);
      }
      
      // / Command for Notion-like slash commands
      else if (e.key === '/' && contentTextarea.selectionStart === contentTextarea.value.lastIndexOf('\n') + 1) {
        e.preventDefault();
        showSlashCommandMenu();
      }
      
      // Cmd/Ctrl + P: Preview
      else if (cmdOrCtrl && e.key === 'p') {
        e.preventDefault();
        togglePreview();
      }
      
      // Cmd/Ctrl + Enter: Save
      else if (cmdOrCtrl && e.key === 'Enter') {
        e.preventDefault();
        postEditor.submit();
      }
    });
    
    // Create keyboard shortcuts guide button
    const shortcutsBtn = document.createElement('button');
    shortcutsBtn.id = 'shortcuts-btn';
    shortcutsBtn.className = 'editor-btn';
    shortcutsBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M14 5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h12zM2 4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2H2z"/><path d="M13 10.25a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5zm0-2a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5zm-5 0A.25.25 0 0 1 8.25 8h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 8 8.75v-.5zm2 0a.25.25 0 0 1 .25-.25h1.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-1.5a.25.25 0 0 1-.25-.25v-.5zm1 2a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5zm-5-2A.25.25 0 0 1 6.25 8h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 6 8.75v-.5zm-2 0A.25.25 0 0 1 4.25 8h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 4 8.75v-.5zm-2 0A.25.25 0 0 1 2.25 8h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 2 8.75v-.5zm11-2a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5zm-2 0a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5zm-2 0A.25.25 0 0 1 9.25 6h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 9 6.75v-.5zm-2 0A.25.25 0 0 1 7.25 6h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 7 6.75v-.5zm-2 0A.25.25 0 0 1 5.25 6h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5A.25.25 0 0 1 5 6.75v-.5zm-3 0A.25.25 0 0 1 2.25 6h1.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-1.5A.25.25 0 0 1 2 6.75v-.5zm0 4a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5zm0-2a.25.25 0 0 1 .25-.25h.5a.25.25 0 0 1 .25.25v.5a.25.25 0 0 1-.25.25h-.5a.25.25 0 0 1-.25-.25v-.5z"/></svg>';
    shortcutsBtn.title = 'Keyboard Shortcuts';
    
    // Append to editor action bar
    if (editorActionBar) {
      const actionBarLeft = editorActionBar.querySelector('.action-bar-left') || editorActionBar;
      actionBarLeft.appendChild(shortcutsBtn);
    }
    
    // Add click event
    shortcutsBtn.addEventListener('click', function() {
      showShortcutsModal();
    });
  }
  
  /**
   * Shows a modal with keyboard shortcuts
   */
  function showShortcutsModal() {
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const cmdKey = isMac ? '⌘' : 'Ctrl';
    
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'shortcuts-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <h3>Keyboard Shortcuts</h3>
        <div class="shortcuts-list">
          <div class="shortcut-group">
            <h4>Text Formatting</h4>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + B</span>
              <span class="shortcut-desc">Bold</span>
            </div>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + I</span>
              <span class="shortcut-desc">Italic</span>
            </div>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + K</span>
              <span class="shortcut-desc">Link</span>
            </div>
          </div>
          <div class="shortcut-group">
            <h4>Headings</h4>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + H, 1</span>
              <span class="shortcut-desc">Heading 1</span>
            </div>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + H, 2</span>
              <span class="shortcut-desc">Heading 2</span>
            </div>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + H, 3</span>
              <span class="shortcut-desc">Heading 3</span>
            </div>
          </div>
          <div class="shortcut-group">
            <h4>Controls</h4>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + P</span>
              <span class="shortcut-desc">Toggle Preview</span>
            </div>
            <div class="shortcut-item">
              <span class="shortcut-keys">${cmdKey} + Enter</span>
              <span class="shortcut-desc">Save Post</span>
            </div>
            <div class="shortcut-item">
              <span class="shortcut-keys">/</span>
              <span class="shortcut-desc">Slash Commands</span>
            </div>
          </div>
        </div>
        <div class="modal-actions">
          <button id="close-shortcuts-btn" class="secondary-btn">Close</button>
        </div>
      </div>
    `;
    
    // Add to body
    document.body.appendChild(modal);
    
    // Show modal
    setTimeout(() => {
      modal.classList.add('active');
      
      // Add close button event
      const closeBtn = document.getElementById('close-shortcuts-btn');
      if (closeBtn) {
        closeBtn.addEventListener('click', function() {
          closeModal(modal);
        });
      }
      
      // Close on click outside
      modal.addEventListener('click', function(e) {
        if (e.target === modal) {
          closeModal(modal);
        }
      });
      
      // Close on Escape key
      document.addEventListener('keydown', function escHandler(e) {
        if (e.key === 'Escape') {
          closeModal(modal);
          document.removeEventListener('keydown', escHandler);
        }
      });
    }, 10);
  }
  
  /**
   * Shows the slash command menu for Notion-like editing
   */
  function showSlashCommandMenu() {
    // Create and position the menu
    const menuContainer = document.createElement('div');
    menuContainer.className = 'slash-command-menu';
    
    // Get cursor position
    const cursorPos = getCursorPosition(contentTextarea);
    
    // Add menu items
    menuContainer.innerHTML = `
      <div class="slash-command-item" data-command="heading1">
        <span class="command-icon">H1</span>
        <span class="command-name">Heading 1</span>
      </div>
      <div class="slash-command-item" data-command="heading2">
        <span class="command-icon">H2</span>
        <span class="command-name">Heading 2</span>
      </div>
      <div class="slash-command-item" data-command="heading3">
        <span class="command-icon">H3</span>
        <span class="command-name">Heading 3</span>
      </div>
      <div class="slash-command-item" data-command="bullet">
        <span class="command-icon">•</span>
        <span class="command-name">Bullet List</span>
      </div>
      <div class="slash-command-item" data-command="numbered">
        <span class="command-icon">1.</span>
        <span class="command-name">Numbered List</span>
      </div>
      <div class="slash-command-item" data-command="quote">
        <span class="command-icon">❝</span>
        <span class="command-name">Quote</span>
      </div>
      <div class="slash-command-item" data-command="code">
        <span class="command-icon">{ }</span>
        <span class="command-name">Code</span>
      </div>
      <div class="slash-command-item" data-command="hr">
        <span class="command-icon">—</span>
        <span class="command-name">Divider</span>
      </div>
      <div class="slash-command-item" data-command="image">
        <span class="command-icon">🖼️</span>
        <span class="command-name">Image</span>
      </div>
    `;
    
    // Position the menu
    document.body.appendChild(menuContainer);
    
    // Position relative to cursor
    const editorRect = contentTextarea.getBoundingClientRect();
    const cursorCoords = getCaretCoordinates(contentTextarea, contentTextarea.selectionStart);
    
    menuContainer.style.position = 'absolute';
    menuContainer.style.top = `${editorRect.top + cursorCoords.top + 20}px`;
    menuContainer.style.left = `${editorRect.left + cursorCoords.left}px`;
    
    // Show menu with animation
    setTimeout(() => {
      menuContainer.classList.add('active');
    }, 10);
    
    // Handle menu item clicks
    const menuItems = menuContainer.querySelectorAll('.slash-command-item');
    menuItems.forEach(item => {
      item.addEventListener('click', function() {
        const command = this.getAttribute('data-command');
        executeSlashCommand(command);
        removeSlashCommandMenu();
      });
    });
    
    // Close menu when clicking outside or pressing Escape
    document.addEventListener('click', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    
    function handleClickOutside(e) {
      if (!menuContainer.contains(e.target) && e.target !== menuContainer) {
        removeSlashCommandMenu();
      }
    }
    
    function handleKeyDown(e) {
      if (e.key === 'Escape') {
        removeSlashCommandMenu();
      }
    }
    
    function removeSlashCommandMenu() {
      document.removeEventListener('click', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
      menuContainer.classList.remove('active');
      setTimeout(() => {
        menuContainer.remove();
      }, 300);
      
      // Remove the slash character
      const text = contentTextarea.value;
      const cursorPos = contentTextarea.selectionStart;
      const textBefore = text.substring(0, cursorPos - 1);
      const textAfter = text.substring(cursorPos);
      contentTextarea.value = textBefore + textAfter;
      contentTextarea.selectionStart = textBefore.length;
      contentTextarea.selectionEnd = textBefore.length;
    }
  }
  
  /**
   * Execute a slash command
   */
  function executeSlashCommand(command) {
    const cursorPos = contentTextarea.selectionStart;
    const text = contentTextarea.value;
    const textBefore = text.substring(0, cursorPos - 1); // -1 to remove the /
    const textAfter = text.substring(cursorPos);
    let newText = '';
    
    switch (command) {
      case 'heading1':
        newText = textBefore + '# ' + textAfter;
        break;
      case 'heading2':
        newText = textBefore + '## ' + textAfter;
        break;
      case 'heading3':
        newText = textBefore + '### ' + textAfter;
        break;
      case 'bullet':
        newText = textBefore + '- ' + textAfter;
        break;
      case 'numbered':
        newText = textBefore + '1. ' + textAfter;
        break;
      case 'quote':
        newText = textBefore + '> ' + textAfter;
        break;
      case 'code':
        newText = textBefore + '```\n' + textAfter + '\n```';
        break;
      case 'hr':
        newText = textBefore + '\n---\n' + textAfter;
        break;
      case 'image':
        // Trigger image upload
        document.getElementById('image-upload-btn').click();
        newText = textBefore + textAfter;
        break;
      default:
        newText = text;
    }
    
    contentTextarea.value = newText;
    contentTextarea.selectionStart = contentTextarea.selectionEnd = 
      command === 'code' ? textBefore.length + 4 : textBefore.length + 2;
    
    // Trigger input event for preview update
    const event = new Event('input');
    contentTextarea.dispatchEvent(event);
  }
  
  /**
   * Format selected text or insert formatting at cursor position
   */
  function formatText(type, param) {
    const start = contentTextarea.selectionStart;
    const end = contentTextarea.selectionEnd;
    const text = contentTextarea.value;
    const selectedText = text.substring(start, end);
    
    let formattedText = '';
    let cursorOffset = 0;
    
    switch (type) {
      case 'bold':
        formattedText = `**${selectedText}**`;
        cursorOffset = 2;
        break;
      case 'italic':
        formattedText = `*${selectedText}*`;
        cursorOffset = 1;
        break;
      case 'link':
        formattedText = selectedText.length > 0 
          ? `[${selectedText}](url)` 
          : `[link text](url)`;
        cursorOffset = selectedText.length > 0 ? 3 : 1;
        break;
      case 'heading':
        const hashes = '#'.repeat(param);
        formattedText = `${hashes} ${selectedText}`;
        cursorOffset = param + 1;
        break;
    }
    
    // Insert the formatted text
    contentTextarea.value = text.substring(0, start) + formattedText + text.substring(end);
    
    // Set cursor position
    if (selectedText.length > 0) {
      contentTextarea.selectionStart = contentTextarea.selectionEnd = start + formattedText.length;
    } else {
      // Place cursor inside the formatting marks
      const cursorPos = start + cursorOffset;
      contentTextarea.selectionStart = contentTextarea.selectionEnd = cursorPos;
    }
    
    // Focus the textarea
    contentTextarea.focus();
    
    // Trigger input event for preview update
    const event = new Event('input');
    contentTextarea.dispatchEvent(event);
  }
  
  /**
   * Get the position of the cursor for the slash command menu
   */
  function getCursorPosition(textarea) {
    // Create a clone of the textarea
    const clone = textarea.cloneNode(true);
    const styles = window.getComputedStyle(textarea);
    
    // Apply styles to make the clone position match exactly
    Object.keys(styles).forEach(key => {
      if (!isNaN(key)) {
        clone.style[styles[key]] = styles.getPropertyValue(styles[key]);
      }
    });
    
    // Position absolutely but keep same dimensions
    clone.style.position = 'absolute';
    clone.style.top = `${textarea.offsetTop}px`;
    clone.style.left = `${textarea.offsetLeft}px`;
    clone.style.width = `${textarea.offsetWidth}px`;
    clone.style.height = `${textarea.offsetHeight}px`;
    clone.style.visibility = 'hidden';
    
    // Add to document briefly
    document.body.appendChild(clone);
    
    // Create a span at the cursor position
    const text = textarea.value;
    const caretPos = textarea.selectionStart;
    clone.value = text.substring(0, caretPos) + '|' + text.substring(caretPos);
    
    // Create a range to find position
    const range = document.createRange();
    const textNode = clone.firstChild;
    
    // Cleanup
    document.body.removeChild(clone);
    
    // Return the position
    return {
      top: textarea.offsetTop + textarea.scrollTop,
      left: textarea.offsetLeft
    };
  }
  
  /**
   * Utility function to get precise caret coordinates
   * Based on GitHub project: https://github.com/component/textarea-caret-position
   */
  function getCaretCoordinates(element, position) {
    // Default position
    const coordinates = {
      top: 15,
      left: 15 
    };
    
    // If browser supports it, use built-in methods
    if (typeof window.getSelection !== 'undefined') {
      try {
        // Create a range at the current cursor position
        const range = document.createRange();
        range.setStart(element, position);
        range.setEnd(element, position);
        
        // Get bounding client rect
        const rect = range.getBoundingClientRect();
        if (rect) {
          coordinates.top = rect.top - element.getBoundingClientRect().top + element.scrollTop;
          coordinates.left = rect.left - element.getBoundingClientRect().left + element.scrollLeft;
        }
      } catch (e) {
        console.error('Error getting caret coordinates:', e);
      }
    }
    
    return coordinates;
  }
  
  /**
   * Insert text at cursor position
   */
  function insertAtCursor(textarea, text) {
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const currentText = textarea.value;
    
    textarea.value = currentText.substring(0, start) + text + currentText.substring(end);
    textarea.selectionStart = textarea.selectionEnd = start + text.length;
    textarea.focus();
  }
  
  /**
   * Set automatic date in the headers
   */
  function setAutomaticDate() {
    // If we're on a new post, set today's date automatically
    const today = new Date().toISOString().split('T')[0];
    dateField.value = today;
    
    // Update the headers textarea with today's date if it's empty
    if (headersTextarea && !headersTextarea.value) {
      headersTextarea.value = `title: 
date: ${today}
tags: 
subtags: 
___`;
    } else if (headersTextarea) {
      // Replace the date line with today's date
      const lines = headersTextarea.value.split('\n');
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim().startsWith('date:')) {
          lines[i] = `date: ${today}`;
          break;
        }
      }
      headersTextarea.value = lines.join('\n');
    }
  }
  
  /**
   * Make title field prominent
   */
  function highlightTitleField() {
    // Find the title field in the headers and add a visual indicator
    if (headersTextarea) {
      // Add styles to highlight title field
      const style = document.createElement('style');
      style.textContent = `
        .title-field-required::before {
          content: "* ";
          color: #e53e3e;
          font-weight: bold;
          position: absolute;
          left: 0;
          top: 0;
        }
        
        .headers-title-field {
          font-weight: bold;
          font-size: 1.1em;
          color: var(--primary-color);
          position: relative;
          padding-left: 15px;
        }
      `;
      document.head.appendChild(style);
      
      // Wrap the title field in headers with a span for styling
      const titleLabel = document.querySelector('label[for="post-headers-textarea"]');
      if (titleLabel) {
        titleLabel.innerHTML = 'Post Headers <small>(title field is required)</small>';
      }
      
      // Add validation on form submit
      postEditor.addEventListener('submit', function(e) {
        // Check if title is provided
        const headersText = headersTextarea.value;
        const titleMatch = headersText.match(/title:\s*(.+)/);
        
        if (!titleMatch || !titleMatch[1].trim()) {
          e.preventDefault();
          alert('Please provide a title for your post.');
          headersTextarea.focus();
        }
      });
    }
  }
  
  /**
   * Mark tags and subtags as optional fields
   */
  function markOptionalFields() {
    // Add visual indicators
    if (headersTextarea) {
      // Add CSS styles
      const style = document.createElement('style');
      style.textContent = `
        .optional-field {
          color: var(--secondary-text);
          font-style: italic;
        }
      `;
      document.head.appendChild(style);
      
      // Add a note to the headers label
      const headersLabel = document.querySelector('label[for="post-headers-textarea"]');
      if (headersLabel) {
        headersLabel.innerHTML += ' <small class="optional-field">(tags and subtags are optional)</small>';
      }
    }
  }
  
  /**
   * Toggle preview mode
   */
  function togglePreview() {
    if (previewSection.classList.contains('hidden')) {
      // Show preview
      updatePreview();
      previewSection.classList.remove('hidden');
      previewButton.textContent = 'Hide Preview';
    } else {
      // Hide preview
      previewSection.classList.add('hidden');
      previewButton.textContent = 'Preview';
    }
  }
  
  /**
   * Update preview content
   */
  function updatePreview() {
    // Get content
    const content = contentTextarea.value;
    
    // Parse content as Markdown (using a basic implementation)
    // In a real app, you would use a full Markdown parser library
    let html = parseMarkdown(content);
    
    // Update preview content
    previewContent.innerHTML = html;
  }
  
  /**
   * Parse headers textarea and update hidden fields
   */
  function parseHeadersToFields() {
    const headersText = headersTextarea.value;
    const lines = headersText.split('\n');
    
    let title = '';
    let date = '';
    let tags = '';
    let subtags = '';
    
    for (const line of lines) {
      // Stop at separator line
      if (line.trim() === '___') {
        break;
      }
      
      // Parse key-value pairs
      const match = line.match(/^([^:]+):\s*(.*)$/);
      if (match) {
        const key = match[1].trim().toLowerCase();
        const value = match[2].trim();
        
        switch (key) {
          case 'title':
            title = value;
            break;
          case 'date':
            date = formatDate(value);
            break;
          case 'tags':
            tags = value;
            break;
          case 'subtags':
            subtags = value;
            break;
        }
      }
    }
    
    // Update hidden fields
    titleField.value = title;
    dateField.value = date;
    tagsField.value = tags;
    subtagsField.value = subtags;
  }
  
  /**
   * Populate headers textarea from individual fields
   */
  function populateHeadersFromFields() {
    const title = titleField.value || '';
    const date = dateField.value || '';
    const tags = tagsField.value || '';
    const subtags = subtagsField.value || '';
    
    let headersText = `title: ${title}\ndate: ${date}`;
    
    if (tags) {
      headersText += `\ntags: ${tags}`;
    }
    
    if (subtags) {
      headersText += `\nsubtags: ${subtags}`;
    }
    
    headersText += '\n___\n';
    headersTextarea.value = headersText;
  }
  
  /**
   * Format date string to YYYY-MM-DD
   * @param {string} dateStr - Date string
   * @returns {string} Formatted date
   */
  function formatDate(dateStr) {
    try {
      const date = new Date(dateStr);
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    } catch (e) {
      console.error('Error formatting date:', e);
      return dateStr;
    }
  }
  
  /**
   * Basic Markdown parser (simplified for demo purposes)
   * In a real app, you would use a library like marked.js
   * @param {string} markdown - Markdown content
   * @returns {string} HTML content
   */
  function parseMarkdown(markdown) {
    if (!markdown) return '';
    
    let html = markdown;
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Bold and Italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Lists
    html = html.replace(/^\s*- (.*$)/gim, '<ul><li>$1</li></ul>');
    html = html.replace(/<\/ul><ul>/g, '');
    
    // Links
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>');
    
    // Images - support for image with size attributes
    html = html.replace(/<img src="(.*?)" alt="(.*?)"(.*)>/g, (match, src, alt, attrs) => {
      return `<img src="${src}" alt="${alt}" ${attrs}>`;
    });
    
    // For standard markdown image syntax
    html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img alt="$1" src="$2">');
    
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Blockquotes
    html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');
    html = html.replace(/<\/blockquote><blockquote>/g, '<br>');
    
    // Paragraphs
    html = html.replace(/\n\s*\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    
    return html;
  }
  
  /**
   * Override admin-panel.js methods to support Bear Blog style editor
   */
  function overrideMethods() {
    // Store original functions
    if (window.editDraft) {
      const originalEditDraft = window.editDraft;
      
      // Override editDraft function
      window.editDraft = function(draftId) {
        // Call original function to maintain compatibility
        originalEditDraft(draftId);
        
        // Get draft data
        const draft = dataStorage.getDrafts().find(d => d.id === draftId);
        if (!draft) return;
        
        // Fill Bear Blog style headers
        if (headersTextarea) {
          let headersText = `title: ${draft.title || ''}
date: ${draft.date || ''}`;
          
          if (draft.tags && draft.tags.length > 0) {
            headersText += `
tags: ${draft.tags.join(', ')}`;
          }
          
          if (draft.subtags && draft.subtags.length > 0) {
            headersText += `
subtags: ${draft.subtags.join(', ')}`;
          }
          
          headersText += `
___`;
          
          headersTextarea.value = headersText;
        }
      };
    }
    
    if (window.editPost) {
      const originalEditPost = window.editPost;
      
      // Override editPost function
      window.editPost = function(postId) {
        // Call original function to maintain compatibility
        originalEditPost(postId);
        
        // Get post data
        const post = dataStorage.getPosts().find(p => p.id === postId);
        if (!post) return;
        
        // Fill Bear Blog style headers
        if (headersTextarea) {
          let headersText = `title: ${post.title || ''}
date: ${post.date || ''}`;
          
          if (post.tags && post.tags.length > 0) {
            headersText += `
tags: ${post.tags.join(', ')}`;
          }
          
          if (post.subtags && post.subtags.length > 0) {
            headersText += `
subtags: ${post.subtags.join(', ')}`;
          }
          
          headersText += `
___`;
          
          headersTextarea.value = headersText;
        }
      };
    }
    
    if (window.showNewPostForm) {
      const originalShowNewPostForm = window.showNewPostForm;
      
      // Override showNewPostForm function
      window.showNewPostForm = function() {
        // Call original function to maintain compatibility
        originalShowNewPostForm();
        
        // Set today's date automatically
        setAutomaticDate();
      };
    }
  }
});