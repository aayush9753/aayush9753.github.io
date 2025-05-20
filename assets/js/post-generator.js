/**
 * Post Generator Module for MyCave
 * Handles post HTML file creation when posts are published
 */

class PostGenerator {
  constructor() {
    this.TEMPLATE_PATH = 'post-template.html';
    this.initialized = false;
    this.templateHtml = '';
    
    // Initialize
    this.init();
  }
  
  /**
   * Initialize the post generator
   */
  async init() {
    try {
      // Load the template HTML
      const response = await fetch(this.TEMPLATE_PATH);
      this.templateHtml = await response.text();
      this.initialized = true;
      
      // Once initialized, check if there are posts that need HTML files
      this.checkPostsForGeneration();
    } catch (error) {
      console.error('Failed to load post template:', error);
    }
  }
  
  /**
   * Check if any posts need HTML files generated
   */
  checkPostsForGeneration() {
    if (!this.initialized || typeof dataStorage === 'undefined') {
      return;
    }
    
    // Get posts from storage
    const posts = dataStorage.getPosts();
    
    // For each post, generate HTML if needed
    posts.forEach(post => {
      // In a real server-side implementation, this would check if the file exists
      // Since we can't write files directly from JavaScript in the browser,
      // this is just a demonstration
      
      // Generate HTML content
      const html = this.generatePostHtml(post);
      
      // In a real implementation, we would save this to a file
      console.log(`Generated HTML for post: ${post.title}`);
      
      // Store the generated HTML in localStorage for demo purposes
      localStorage.setItem(`post-html-${post.id}`, html);
    });
  }
  
  /**
   * Generate HTML for a post
   */
  generatePostHtml(post) {
    if (!this.templateHtml) {
      return null;
    }
    
    // Parse template into a DOM we can safely manipulate
    const parser = new DOMParser();
    const doc = parser.parseFromString(this.templateHtml, 'text/html');

    if (!doc) return null;

    // <title>
    doc.title = `${post.title} - MyCave`;

    // <meta description>
    const metaDesc = doc.querySelector('meta[name="description"]');
    if (metaDesc) {
      // Create a safe preview by removing markdown characters and limiting length
      const plainText = post.content
        .replace(/[#*`_]/g, '') // Remove markdown formatting characters
        .replace(/\[.*?\]\(.*?\)/g, '$1'); // Replace markdown links with just the text
      
      const description = plainText.length > 150 ? plainText.substring(0, 147) + '...' : plainText;
      metaDesc.setAttribute('content', description);
    }

    // Article elements
    const article = doc.querySelector('article');
    if (article) {
      // Title inside post header
      const headerTitle = article.querySelector('.post-header h1');
      if (headerTitle) headerTitle.textContent = post.title;

      // Date
      const dateSpan = article.querySelector('.post-header .date');
      if (dateSpan) dateSpan.textContent = post.date;

      // Tags / Subtags
      const tagsContainer = article.querySelector('.post-header .tags');
      if (tagsContainer) {
        tagsContainer.innerHTML = '';
        post.tags.forEach(tag => {
          const span = doc.createElement('span');
          span.className = 'tag';
          span.textContent = tag;
          tagsContainer.appendChild(span);
        });
        if (post.subtags) {
          post.subtags.forEach(sub => {
            const span = doc.createElement('span');
            span.className = 'subtag';
            span.textContent = sub;
            tagsContainer.appendChild(span);
          });
        }
      }

      // Content
      const contentDiv = article.querySelector('.post-content');
      if (contentDiv) {
        const contentHtml = typeof markdownRenderer !== 'undefined'
          ? markdownRenderer.render(post.content)
          : post.content;
        contentDiv.innerHTML = contentHtml;
      }
    }

    // Serialize back to HTML string
    return '<!DOCTYPE html>' + doc.documentElement.outerHTML;
  }
  
  /**
   * Method to generate a post HTML when a new post is published
   * In a real implementation, this would write to a file
   */
  generatePostFileForPost(post) {
    if (!this.initialized) {
      console.error('Post generator not initialized');
      return false;
    }
    
    const html = this.generatePostHtml(post);
    
    if (html) {
      // In a real server-side implementation, this would write to a file
      // For demonstration, store in localStorage
      localStorage.setItem(`post-html-${post.id}`, html);
      
      // Also store the URL to post mapping for easier navigation
      const postUrls = JSON.parse(localStorage.getItem('post-urls') || '{}');
      postUrls[post.url] = post.id;
      localStorage.setItem('post-urls', JSON.stringify(postUrls));
      
      console.log(`Generated HTML for post: ${post.title} with URL: ${post.url}`);
      return true;
    }
    
    return false;
  }
}

// Create a global instance
const postGenerator = new PostGenerator();

// Add event listener to handle post creation
document.addEventListener('DOMContentLoaded', function() {
  // Override dataStorage.addPost to generate HTML when a post is added
  const originalAddPost = dataStorage.addPost;
  
  dataStorage.addPost = function(post) {
    // Call original method
    const result = originalAddPost.call(this, post);
    
    // Generate HTML
    if (result && typeof postGenerator !== 'undefined') {
      postGenerator.generatePostFileForPost(result);
    }
    
    return result;
  };
  
  // Override dataStorage.updatePost to regenerate HTML when a post is updated
  const originalUpdatePost = dataStorage.updatePost;
  
  dataStorage.updatePost = function(postId, updatedPost) {
    // Call original method
    const result = originalUpdatePost.call(this, postId, updatedPost);
    
    // Regenerate HTML
    if (result && typeof postGenerator !== 'undefined') {
      postGenerator.generatePostFileForPost(result);
    }
    
    return result;
  };
  
  // Override dataStorage.publishDraft to generate HTML when a draft is published
  const originalPublishDraft = dataStorage.publishDraft;
  
  dataStorage.publishDraft = function(draftId) {
    // Call original method
    const result = originalPublishDraft.call(this, draftId);
    
    // Generate HTML
    if (result && typeof postGenerator !== 'undefined') {
      postGenerator.generatePostFileForPost(result);
    }
    
    return result;
  };
});