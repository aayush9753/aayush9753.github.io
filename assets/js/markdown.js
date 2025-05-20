/**
 * Simple Markdown Renderer
 * Converts Markdown syntax to HTML for MyCave
 */

class MarkdownRenderer {
  constructor() {
    this.rules = [
      // Headers
      { pattern: /^# (.+)$/gm, replacement: '<h1>$1</h1>' },
      { pattern: /^## (.+)$/gm, replacement: '<h2>$1</h2>' },
      { pattern: /^### (.+)$/gm, replacement: '<h3>$1</h3>' },
      { pattern: /^#### (.+)$/gm, replacement: '<h4>$1</h4>' },
      { pattern: /^##### (.+)$/gm, replacement: '<h5>$1</h5>' },
      { pattern: /^###### (.+)$/gm, replacement: '<h6>$1</h6>' },
      
      // Bold and Italic
      { pattern: /\*\*(.+?)\*\*/g, replacement: '<strong>$1</strong>' },
      { pattern: /\*(.+?)\*/g, replacement: '<em>$1</em>' },
      { pattern: /_(.+?)_/g, replacement: '<em>$1</em>' },
      
      // Links
      { pattern: /\[(.+?)\]\((.+?)\)/g, replacement: '<a href="$2">$1</a>' },
      
      // Images
      { pattern: /!\[(.+?)\]\((.+?)\)/g, replacement: '<img src="$2" alt="$1">' },
      
      // Blockquotes
      { pattern: /^> (.+)$/gm, replacement: '<blockquote><p>$1</p></blockquote>' },
      
      // Lists
      // Ordered lists need special handling
      // Unordered lists need special handling
      
      // Code blocks (multiline)
      { pattern: /```([\s\S]+?)```/g, replacement: '<pre><code>$1</code></pre>' },
      { pattern: /`([^`]+)`/g, replacement: '<code>$1</code>' },
      
      // Horizontal rule
      { pattern: /^---$/gm, replacement: '<hr>' },
      
      // Paragraphs (must be last)
      { pattern: /^([^<].*?)$/gm, replacement: function(match) {
        // Skip if it's already wrapped in a tag
        if (match.trim() === '' || match.startsWith('<')) return match;
        return '<p>' + match + '</p>';
      } }
    ];
  }
  
  /**
   * Process Markdown text through all rules
   */
  render(markdown) {
    let html = markdown;
    
    // Special handling for lists before other processing
    html = this.processList(html);
    
    // Apply all other rules
    for (const rule of this.rules) {
      if (typeof rule.replacement === 'function') {
        html = html.replace(rule.pattern, rule.replacement);
      } else {
        html = html.replace(rule.pattern, rule.replacement);
      }
    }
    
    // Replace newlines with breaks for remaining newlines
    html = html.replace(/\n/g, '<br>');
    
    return html;
  }
  
  /**
   * Special handler for list items
   */
  processList(markdown) {
    // Process unordered lists - add a data attribute to identify the list type
    let html = markdown.replace(/^[ \t]*[-*+] (.+)$/gm, '<li data-list-type="ul">$1</li>');
    
    // Process ordered lists - add a data attribute to identify the list type
    html = html.replace(/^[ \t]*(\d+)\. (.+)$/gm, '<li data-list-type="ol">$2</li>');
    
    // Find consecutive list items and group them
    const lines = html.split('\n');
    let result = [];
    let currentListType = null;
    let currentListItems = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const liMatch = line.match(/<li data-list-type=["'](ol|ul)["']/);
      
      if (liMatch) {
        const listType = liMatch[1];
        
        // If this is the first list item or we're switching list types
        if (currentListType === null || currentListType !== listType) {
          // If we have accumulated list items, wrap and add them
          if (currentListItems.length > 0) {
            result.push(`<${currentListType}>${currentListItems.join('')}</${currentListType}>`);
            currentListItems = [];
          }
          
          // Start a new list
          currentListType = listType;
        }
        
        // Add the item (but remove the data attribute)
        currentListItems.push(line.replace(/data-list-type=["'](ol|ul)["'] ?/g, ''));
      } else {
        // If we have accumulated list items, wrap and add them
        if (currentListItems.length > 0) {
          result.push(`<${currentListType}>${currentListItems.join('')}</${currentListType}>`);
          currentListItems = [];
          currentListType = null;
        }
        
        // Add the non-list line
        result.push(line);
      }
    }
    
    // Handle any remaining list items
    if (currentListItems.length > 0) {
      result.push(`<${currentListType}>${currentListItems.join('')}</${currentListType}>`);
    }
    
    return result.join('\n');
  }
}

// Global instance for use in the application
const markdownRenderer = new MarkdownRenderer();

// Add to post editor if in admin mode
document.addEventListener('DOMContentLoaded', function() {
  const postContent = document.getElementById('post-content');
  const postForm = document.getElementById('post-form');
  
  // If we're on the admin page with the form
  if (postContent && postForm) {
    // Create a preview area
    const previewArea = document.createElement('div');
    previewArea.id = 'markdown-preview';
    previewArea.className = 'markdown-preview';
    previewArea.innerHTML = '<h4>Content Preview</h4><div class="preview-content"></div>';
    
    // Insert after the editor container
    const editorContainer = document.querySelector('.editor-container');
    if (editorContainer) {
      editorContainer.appendChild(previewArea);
    } else {
      // Fallback to the old method if editor container doesn't exist
      postContent.parentNode.insertBefore(previewArea, postContent.nextSibling);
    }
    
    // Add live preview functionality
    postContent.addEventListener('input', function() {
      const markdown = this.value;
      const html = markdownRenderer.render(markdown);
      document.querySelector('.preview-content').innerHTML = html;
    });
  }
});