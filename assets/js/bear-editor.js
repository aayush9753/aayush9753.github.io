/**
 * Bear Blog Style Editor for MyCave
 * Handles YAML-style metadata headers and Markdown content
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
    
    // Images
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
        
        // Set today's date
        const today = new Date().toISOString().split('T')[0];
        
        // Initialize Bear Blog style headers
        if (headersTextarea) {
          headersTextarea.value = `title: 
date: ${today}
tags: 
subtags: 
___`;
        }
      };
    }
  }
});