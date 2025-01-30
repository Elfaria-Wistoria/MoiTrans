// Initialize animations
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Add loading animation to translate button
    const translateBtn = document.getElementById('translateBtn');
    if (translateBtn) {
        translateBtn.addEventListener('click', function() {
            this.classList.add('animate-pulse');
            setTimeout(() => {
                this.classList.remove('animate-pulse');
            }, 1000);
        });
    }
});

// Add scroll animations
window.addEventListener('scroll', function() {
    const nav = document.querySelector('nav');
    if (window.scrollY > 50) {
        nav.classList.add('shadow-lg');
    } else {
        nav.classList.remove('shadow-lg');
    }
});