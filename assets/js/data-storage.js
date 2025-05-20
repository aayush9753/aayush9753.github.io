/**
 * Data Storage Module for MyCave
 * Handles post and draft storage using localStorage
 */

class DataStorage {
  constructor() {
    this.POSTS_KEY = 'mycave-posts';
    this.DRAFTS_KEY = 'mycave-drafts';
    this.ADMIN_KEY = 'mycave-admin';
    this.posts = [];
    this.drafts = [];
    this.isAdmin = false;
    
    // Initialize with data
    this.init();
  }
  
  /**
   * Initialize the data storage
   */
  init() {
    // Load posts from localStorage or use default data
    const savedPosts = localStorage.getItem(this.POSTS_KEY);
    if (savedPosts) {
      this.posts = JSON.parse(savedPosts);
    } else {
      // Use default data as initial data
      this.posts = [
        {
          id: 'post-1',
          title: 'Notes on building a second brain',
          date: '2023-07-15',
          tags: ['productivity', 'knowledge-management'],
          subtags: ['note-taking', 'zettelkasten'],
          content: 'The concept of a "second brain" has gained significant traction in productivity circles. At its core, it\'s about creating an external system to store, organize, and retrieve the information we consume, helping us think more clearly and create more effectively.',
          url: 'post-template.html'
        },
        {
          id: 'post-2',
          title: 'Reflections on minimalism',
          date: '2023-06-30',
          tags: ['lifestyle', 'philosophy'],
          subtags: ['digital-minimalism', 'simplicity'],
          content: "Minimalism is more than an aesthetic choice; it's a mindset that prioritizes value over volume. By deliberately choosing what to include in our lives, we create space for what truly matters.",
          url: '#'
        },
        {
          id: 'post-3',
          title: 'Understanding plain text productivity',
          date: '2023-06-10',
          tags: ['productivity', 'tools'],
          subtags: ['markdown', 'text-files'],
          content: "Plain text systems offer surprising advantages for productivity: they're portable, future-proof, and distraction-free. By embracing the constraints of simple text files, we can focus on what matters: our thoughts and ideas.",
          url: '#'
        },
        {
          id: 'post-4',
          title: 'Digital gardens vs traditional blogs',
          date: '2023-05-22',
          tags: ['writing', 'web'],
          subtags: ['digital-gardens', 'publishing'],
          content: 'Unlike chronological blogs, digital gardens are non-linear, continuously evolving collections of thoughts and notes. They emphasize connection and growth over temporal organization.',
          url: '#'
        },
        {
          id: 'post-5',
          title: 'Getting started with personal knowledge management',
          date: '2023-05-01',
          tags: ['productivity', 'knowledge-management'],
          subtags: ['note-taking', 'organization'],
          content: 'Personal knowledge management is about systematically capturing, organizing, and sharing what you know and learn. It begins with reliable capture methods and requires consistent maintenance.',
          url: '#'
        }
      ];
      this.savePosts();
    }
    
    // Load drafts from localStorage
    const savedDrafts = localStorage.getItem(this.DRAFTS_KEY);
    if (savedDrafts) {
      this.drafts = JSON.parse(savedDrafts);
    }
    
    try {
      // Check admin status
      const adminStatus = localStorage.getItem(this.ADMIN_KEY);
      this.isAdmin = adminStatus === 'true';
      console.log('Admin status initialized from localStorage:', this.isAdmin);
    } catch (err) {
      console.error('Error loading admin status:', err);
      this.isAdmin = false;
    }
  }
  
  /**
   * Save posts to localStorage
   */
  savePosts() {
    localStorage.setItem(this.POSTS_KEY, JSON.stringify(this.posts));
  }
  
  /**
   * Save drafts to localStorage
   */
  saveDrafts() {
    localStorage.setItem(this.DRAFTS_KEY, JSON.stringify(this.drafts));
  }
  
  /**
   * Get all published posts
   */
  getPosts() {
    return [...this.posts];
  }
  
  /**
   * Get all drafts
   */
  getDrafts() {
    return [...this.drafts];
  }
  
  /**
   * Generate a cryptographically-random ID (with graceful fallback)
   */
  generateId(prefix = 'id') {
    let unique;
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      unique = crypto.randomUUID();
    } else {
      // Fallback: timestamp + random suffix
      unique = Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
    }
    return `${prefix}-${unique}`;
  }

  /**
   * Create a readable yet unique slug from a title
   */
  generateSlug(title) {
    const base = (title || 'post')
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '') // keep alphanum, underscore, dash, space
      .replace(/\s+/g, '-');

    const hash = Date.now().toString(36).slice(-4);
    return `${base}-${hash}`;
  }

  /**
   * Add a new post
   */
  addPost(post) {
    // Generate a stable id if missing
    if (!post.id) {
      post.id = this.generateId('post');
    }

    // Generate a unique-ish URL/slug if missing or placeholder
    if (!post.url || post.url === '#') {
      const slug = this.generateSlug(post.title);
      
      // Check if posts directory exists in localStorage
      const postsDirectoryExists = localStorage.getItem('posts-directory-exists');
      
      if (!postsDirectoryExists) {
        // In a real environment, we would create the directory here
        // For this simulation, we'll just set a flag
        localStorage.setItem('posts-directory-exists', 'true');
      }
      
      // Use the correct path format
      post.url = `${slug}.html`;
    }

    this.posts.push(post);
    this.savePosts();
    return post;
  }
  
  /**
   * Save a draft
   */
  saveDraft(draft) {
    // Generate ID if not provided
    if (!draft.id) {
      draft.id = this.generateId('draft');
    }
    
    // Check if this draft already exists
    const existingIndex = this.drafts.findIndex(d => d.id === draft.id);
    
    if (existingIndex >= 0) {
      // Update existing draft
      this.drafts[existingIndex] = draft;
    } else {
      // Add new draft
      this.drafts.push(draft);
    }
    
    this.saveDrafts();
    return draft;
  }
  
  /**
   * Delete a draft
   */
  deleteDraft(draftId) {
    this.drafts = this.drafts.filter(draft => draft.id !== draftId);
    this.saveDrafts();
  }
  
  /**
   * Publish a draft
   */
  publishDraft(draftId) {
    const draft = this.drafts.find(draft => draft.id === draftId);
    
    if (draft) {
      // Create a new post from the draft
      const post = {
        ...draft,
        id: draft.id.replace('draft-', 'post-')
      };
      
      // Add to posts
      this.addPost(post);
      
      // Remove from drafts
      this.deleteDraft(draftId);
      
      return post;
    }
    
    return null;
  }
  
  /**
   * Update an existing post
   */
  updatePost(postId, updatedPost) {
    const index = this.posts.findIndex(post => post.id === postId);
    
    if (index >= 0) {
      this.posts[index] = {
        ...this.posts[index],
        ...updatedPost,
        id: postId // Ensure ID doesn't change
      };
      
      this.savePosts();
      return this.posts[index];
    }
    
    return null;
  }
  
  /**
   * Delete a post
   */
  deletePost(postId) {
    this.posts = this.posts.filter(post => post.id !== postId);
    this.savePosts();
  }
  
  /**
   * Set admin status
   */
  setAdminStatus(isAdmin) {
    try {
      this.isAdmin = Boolean(isAdmin);
      localStorage.setItem(this.ADMIN_KEY, this.isAdmin.toString());
      console.log('Admin status set to:', this.isAdmin);
    } catch (err) {
      console.error('Error setting admin status:', err);
    }
  }
  
  /**
   * Check if user is admin
   */
  checkAdminStatus() {
    return this.isAdmin;
  }
  
  /**
   * Login as admin
   */
  adminLogin(password) {
    try {
      // Simple password check - in a real app, use secure authentication
      const isCorrect = password === 'admin123'; // Example password
      
      if (isCorrect) {
        console.log('Password correct, setting admin status to true');
        this.setAdminStatus(true);
      } else {
        console.log('Password incorrect:', password);
      }
      
      return isCorrect;
    } catch (err) {
      console.error('Error in admin login:', err);
      return false;
    }
  }
  
  /**
   * Logout from admin
   */
  adminLogout() {
    this.setAdminStatus(false);
  }

  /**
   * Clear admin status from localStorage
   */
  resetAdminStatus() {
    localStorage.removeItem(this.ADMIN_KEY);
    this.isAdmin = false;
    console.log('Admin status reset');
  }
}

// Create and export the dataStorage instance
window.dataStorage = new DataStorage();