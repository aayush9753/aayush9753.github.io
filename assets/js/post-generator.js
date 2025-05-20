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
    
    // Replace template placeholders with post data
    let html = this.templateHtml;
    
    // Title
    html = html.replace(/<title>.*?<\/title>/, `<title>${post.title} - MyCave</title>`);
    
    // Meta description
    const description = post.content.length > 150 ? post.content.substring(0, 147) + '...' : post.content;
    html = html.replace(/<meta name="description" content=".*?">/, `<meta name="description" content="${description}">`);
    
    // Post header
    html = html.replace(/<h1>.*?<\/h1>/, `<h1>${post.title}</h1>`);
    
    // Post meta
    const tagsHtml = post.tags.map(tag => `<span class="tag">${tag}</span>`).join('\n            ');
    const subtagsHtml = post.subtags ? post.subtags.map(subtag => `<span class="subtag">${subtag}</span>`).join('\n            ') : '';
    
    html = html.replace(
      /<span class="date">.*?<\/span>/,
      `<span class="date">${post.date}</span>`
    );
    
    html = html.replace(
      /<div class="tags">[\s\S]*?<\/div>/,
      `<div class="tags">
            ${tagsHtml}
            ${subtagsHtml}
          </div>`
    );
    
    // Post content
    // Use markdown renderer to convert content to HTML
    const contentHtml = typeof markdownRenderer !== 'undefined' 
      ? markdownRenderer.render(post.content)
      : post.content;
    
    html = html.replace(
      /<div class="post-content">[\s\S]*?<\/div>/,
      `<div class="post-content">
          ${contentHtml}
        </div>`
    );
    
    // Previous/Next posts
    // This would normally be dynamic based on post dates
    
    return html;
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
      console.log(`Generated HTML for post: ${post.title}`);
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