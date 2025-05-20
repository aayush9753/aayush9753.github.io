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
      
      // Code blocks
      { pattern: /```([^`]+)```/g, replacement: '<pre><code>$1</code></pre>' },
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
    // Process unordered lists
    let html = markdown.replace(/^[ \t]*[-*+] (.+)$/gm, '<li>$1</li>');
    
    // Wrap adjacent list items in ul tags
    html = html.replace(/(<li>.+?<\/li>)\n(?=<li>)/g, '$1');
    html = html.replace(/(<li>.+<\/li>)/g, '<ul>$1</ul>');
    
    // Process ordered lists
    html = html.replace(/^[ \t]*(\d+)\. (.+)$/gm, '<li>$2</li>');
    
    // Wrap adjacent ordered list items in ol tags
    html = html.replace(/(<li>.+?<\/li>)\n(?=<li>)/g, '$1');
    html = html.replace(/(<li>.+<\/li>)/g, function(match) {
      // Check if it's part of an ordered list
      if (match.match(/\d+\./)) {
        return '<ol>' + match + '</ol>';
      }
      return match;
    });
    
    return html;
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
    
    // Insert after the content textarea
    postContent.parentNode.insertBefore(previewArea, postContent.nextSibling);
    
    // Add live preview functionality
    postContent.addEventListener('input', function() {
      const markdown = this.value;
      const html = markdownRenderer.render(markdown);
      document.querySelector('.preview-content').innerHTML = html;
    });
  }
});