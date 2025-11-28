// Image Zoom Functionality
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    const closeBtn = document.querySelector('.modal-close');

    // Add click event to all zoomable images
    const zoomableImages = document.querySelectorAll('.zoomable');

    zoomableImages.forEach(img => {
        img.addEventListener('click', function () {
            modal.style.display = 'block';
            modalImg.src = this.src;
            modalCaption.textContent = this.alt;

            // Prevent body scroll when modal is open
            document.body.style.overflow = 'hidden';
        });
    });

    // Close modal when clicking the X button
    closeBtn.addEventListener('click', closeModal);

    // Close modal when clicking outside the image
    modal.addEventListener('click', function (e) {
        if (e.target === modal || e.target === modalImg) {
            closeModal();
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });

    function closeModal() {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
});
