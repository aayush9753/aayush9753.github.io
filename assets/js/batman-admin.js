/**
 * Batman Secret Admin Module for MyCave
 * Provides a hidden admin access through a Batman logo
 */

document.addEventListener('DOMContentLoaded', function() {
  // Elements
  const batmanAdminBtn = document.getElementById('batman-admin');
  const batmanModal = document.getElementById('batman-admin-modal');
  const batmanPasswordInput = document.getElementById('batman-password');
  const batmanLoginBtn = document.getElementById('batman-login-btn');
  const batmanCloseBtn = document.getElementById('batman-close-btn');
  const adminViewBtn = document.getElementById('admin-view-btn');
  
  // Check if all elements are present
  if (!batmanAdminBtn || !batmanModal || !batmanPasswordInput || !batmanLoginBtn || !batmanCloseBtn) {
    console.error('Batman admin elements not found in the DOM');
    return;
  }
  
  if (!adminViewBtn) {
    console.error('Admin view button not found in the DOM');
    return;
  }
  
  // Hide the regular admin button by default
  adminViewBtn.style.display = 'none';
  
  // Open Batman admin modal
  batmanAdminBtn.addEventListener('click', function() {
    batmanModal.classList.add('active');
    batmanPasswordInput.focus();
  });
  
  // Close the modal
  batmanCloseBtn.addEventListener('click', function() {
    batmanModal.classList.remove('active');
    batmanPasswordInput.value = '';
  });
  
  // Close modal on clicking outside of it
  batmanModal.addEventListener('click', function(e) {
    if (e.target === batmanModal) {
      batmanModal.classList.remove('active');
      batmanPasswordInput.value = '';
    }
  });
  
  // Handle login with Enter key
  batmanPasswordInput.addEventListener('keyup', function(e) {
    if (e.key === 'Enter') {
      verifyBatmanPassword();
    }
  });
  
  // Handle login button click
  batmanLoginBtn.addEventListener('click', verifyBatmanPassword);
  
  /**
   * Verify Batman secret password and activate admin mode
   */
  function verifyBatmanPassword() {
    const enteredPassword = batmanPasswordInput.value;
    
    // Clear the input field for security 
    batmanPasswordInput.value = '';
    
    // Attempt login with entered password
    try {
      const isAdmin = dataStorage.adminLogin(enteredPassword);
      
      if (isAdmin) {
        // Hide Batman modal
        batmanModal.classList.remove('active');
        
        // Activate admin view
        adminViewBtn.click();
        
        // Show a brief confirmation toast (if there's a UI for it)
        showAdminToast('Admin mode activated');
      } else {
        // Show error message
        alert('Incorrect password. Access denied.');
      }
      
    } catch (err) {
      console.error('Error verifying Batman password:', err);
      alert('Something went wrong. Please try again.');
    }
  }
  
  /**
   * Show a toast notification (without requiring additional libraries)
   * @param {string} message - Message to display
   */
  function showAdminToast(message) {
    // Create toast element if it doesn't exist
    let toast = document.getElementById('admin-toast');
    
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'admin-toast';
      toast.style.position = 'fixed';
      toast.style.bottom = '20px';
      toast.style.left = '50%';
      toast.style.transform = 'translateX(-50%)';
      toast.style.background = 'rgba(0, 0, 0, 0.8)';
      toast.style.color = 'white';
      toast.style.padding = '10px 20px';
      toast.style.borderRadius = '4px';
      toast.style.fontSize = '14px';
      toast.style.zIndex = '2000';
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease-in-out';
      document.body.appendChild(toast);
    }
    
    // Set message and show toast
    toast.textContent = message;
    toast.style.opacity = '1';
    
    // Hide toast after 3 seconds
    setTimeout(function() {
      toast.style.opacity = '0';
    }, 3000);
  }
});