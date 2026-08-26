document.addEventListener("DOMContentLoaded", function () {


    /*
    ============================================
    PASSWORD SHOW / HIDE
    ============================================
    */

    const passwordButtons =
        document.querySelectorAll(".password-toggle");


    passwordButtons.forEach(function (button) {

        button.addEventListener("click", function () {

            const targetId =
                button.getAttribute("data-target");

            const passwordInput =
                document.getElementById(targetId);


            if (!passwordInput) {
                return;
            }


            if (passwordInput.type === "password") {

                passwordInput.type = "text";

                button.textContent = "Hide";

            }

            else {

                passwordInput.type = "password";

                button.textContent = "Show";

            }

        });

    });



    /*
    ============================================
    PASSWORD MATCH CHECK
    ============================================
    */

    const registerForm =
        document.querySelector(".register-card form");


    if (registerForm) {

        registerForm.addEventListener(
            "submit",
            function (event) {

                const password =
                    document.getElementById(
                        "registerPassword"
                    );

                const confirmPassword =
                    document.getElementById(
                        "confirmPassword"
                    );


                if (
                    password &&
                    confirmPassword &&
                    password.value !== confirmPassword.value
                ) {

                    event.preventDefault();

                    alert("Passwords do not match.");

                }

            }
        );

    }



    /*
    ============================================
    ACCOUNT DROPDOWN
    ============================================
    */

    const accountButton =
        document.getElementById("accountButton");

    const accountDropdown =
        document.getElementById("accountDropdown");


    if (accountButton && accountDropdown) {


        accountButton.addEventListener(
            "click",
            function (event) {

                event.stopPropagation();

                accountDropdown.classList.toggle("show");

            }
        );


        accountDropdown.addEventListener(
            "click",
            function (event) {

                event.stopPropagation();

            }
        );


        document.addEventListener(
            "click",
            function () {

                accountDropdown.classList.remove("show");

            }
        );


        document.addEventListener(
            "keydown",
            function (event) {

                if (event.key === "Escape") {

                    accountDropdown.classList.remove(
                        "show"
                    );

                }

            }
        );

    }



    /*
    ============================================
    MOBILE MENU
    ============================================
    */

    const menuToggle =
        document.getElementById("menuToggle");

    const navLinks =
        document.getElementById("navLinks");


    if (menuToggle && navLinks) {


        menuToggle.addEventListener(
            "click",
            function () {

                navLinks.classList.toggle("mobile-open");

                menuToggle.classList.toggle("active");

            }
        );


        const mobileNavLinks =
            navLinks.querySelectorAll("a");


        mobileNavLinks.forEach(function (link) {

            link.addEventListener(
                "click",
                function () {

                    navLinks.classList.remove(
                        "mobile-open"
                    );

                    menuToggle.classList.remove(
                        "active"
                    );

                }
            );

        });

    }



    /*
    ============================================
    SMOOTH SCROLL
    ============================================
    */

    const internalLinks =
        document.querySelectorAll(
            'a[href^="#"]'
        );


    internalLinks.forEach(function (link) {

        link.addEventListener(
            "click",
            function (event) {

                const targetId =
                    link.getAttribute("href");


                if (
                    !targetId ||
                    targetId === "#"
                ) {

                    return;

                }


                const target =
                    document.querySelector(targetId);


                if (target) {

                    event.preventDefault();

                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });

                }

            }
        );

    });



    /*
    ============================================
    ACTIVE NAVIGATION LINK
    ============================================
    */

    const sections =
        document.querySelectorAll(
            "main section[id]"
        );

    const navigationLinks =
        document.querySelectorAll(
            ".nav-link"
        );


    if (
        sections.length > 0 &&
        navigationLinks.length > 0
    ) {


        const observer =
            new IntersectionObserver(
                function (entries) {

                    entries.forEach(
                        function (entry) {

                            if (entry.isIntersecting) {

                                const currentId =
                                    entry.target.getAttribute(
                                        "id"
                                    );


                                navigationLinks.forEach(
                                    function (link) {

                                        link.classList.remove(
                                            "active"
                                        );


                                        const href =
                                            link.getAttribute(
                                                "href"
                                            );


                                        if (
                                            href ===
                                            "#" + currentId
                                        ) {

                                            link.classList.add(
                                                "active"
                                            );

                                        }

                                    }
                                );

                            }

                        }
                    );

                },
                {
                    threshold: 0.35
                }
            );


        sections.forEach(
            function (section) {

                observer.observe(section);

            }
        );

    }



    /*
    ============================================
    RETINA IMAGE UPLOAD
    ============================================
    */

    const imageInput =
        document.getElementById(
            "imageInput"
        );

    const browseButton =
        document.getElementById(
            "browseButton"
        );

    const uploadArea =
        document.getElementById(
            "uploadArea"
        );

    const imagePreview =
        document.getElementById(
            "imagePreview"
        );

    const imagePreviewContainer =
        document.getElementById(
            "imagePreviewContainer"
        );


    /*
    BROWSE BUTTON
    */

    if (
        browseButton &&
        imageInput
    ) {

        browseButton.addEventListener(
            "click",
            function () {

                imageInput.click();

            }
        );

    }



    /*
    FILE SELECTED
    */

    if (imageInput) {

        imageInput.addEventListener(
            "change",
            function () {

                const file =
                    imageInput.files[0];


                if (file) {

                    handleImageFile(file);

                }

            }
        );

    }



    /*
    ============================================
    HANDLE IMAGE FILE
    ============================================
    */

    function handleImageFile(file) {


        /*
        CHECK FILE TYPE
        */

        if (
            file.type !== "image/jpeg" &&
            file.type !== "image/png"
        ) {

            alert(
                "Please upload a JPG or PNG retina image."
            );

            return;

        }


        /*
        CHECK FILE SIZE
        */

        const maxSize =
            10 * 1024 * 1024;


        if (file.size > maxSize) {

            alert(
                "Image size must be less than 10MB."
            );

            return;

        }


        /*
        CREATE PREVIEW
        */

        const reader =
            new FileReader();


        reader.onload =
            function (event) {

                if (imagePreview) {

                    imagePreview.src =
                        event.target.result;

                }


                if (imagePreviewContainer) {

                    imagePreviewContainer.classList.add(
                        "show"
                    );

                }


                if (uploadArea) {

                    uploadArea.classList.add(
                        "has-image"
                    );

                }

            };


        reader.readAsDataURL(file);

    }



    /*
    ============================================
    DRAG & DROP
    ============================================
    */

    if (uploadArea) {


        uploadArea.addEventListener(
            "dragover",
            function (event) {

                event.preventDefault();

                uploadArea.classList.add(
                    "dragging"
                );

            }
        );


        uploadArea.addEventListener(
            "dragleave",
            function () {

                uploadArea.classList.remove(
                    "dragging"
                );

            }
        );


        uploadArea.addEventListener(
            "drop",
            function (event) {

                event.preventDefault();

                uploadArea.classList.remove(
                    "dragging"
                );


                const files =
                    event.dataTransfer.files;


                if (files.length > 0) {

                    const file =
                        files[0];


                    handleImageFile(file);


                    /*
                    UPDATE INPUT
                    */

                    if (imageInput) {

                        try {

                            const dataTransfer =
                                new DataTransfer();

                            dataTransfer.items.add(
                                file
                            );

                            imageInput.files =
                                dataTransfer.files;

                        }

                        catch (error) {

                            console.log(
                                "Could not update file input."
                            );

                        }

                    }

                }

            }
        );


        /*
        CLICK UPLOAD AREA
        */

        uploadArea.addEventListener(
            "click",
            function (event) {

                if (
                    event.target.closest(
                        "#browseButton"
                    )
                ) {

                    return;

                }


                if (imageInput) {

                    imageInput.click();

                }

            }
        );

    }



    /*
    ============================================
    ANALYZE IMAGE BUTTON
    ============================================
    */

    const analyzeButton =
        document.getElementById(
            "analyzeButton"
        );

    const resultCard =
        document.getElementById(
            "resultCard"
        );


    if (analyzeButton) {

        analyzeButton.addEventListener(
            "click",
            function () {


                /*
                NO IMAGE
                */

                if (
                    !imageInput ||
                    !imageInput.files ||
                    imageInput.files.length === 0
                ) {

                    alert(
                        "Please upload a retina image first."
                    );

                    return;

                }


                /*
                BUTTON LOADING STATE
                */

                const originalText =
                    analyzeButton.innerHTML;


                analyzeButton.disabled =
                    true;


                analyzeButton.innerHTML =
                    `
                    <span>
                        Analyzing image...
                    </span>

                    <span class="analyze-arrow">
                        ◌
                    </span>
                    `;


                /*
                TEMPORARY DEMO RESULT
                */

                setTimeout(
                    function () {


                        analyzeButton.disabled =
                            false;


                        analyzeButton.innerHTML =
                            originalText;


                        if (resultCard) {

                            resultCard.classList.add(
                                "show"
                            );


                            resultCard.scrollIntoView({
                                behavior: "smooth",
                                block: "center"
                            });

                        }

                    },
                    1800
                );

            }
        );

    }



    /*
    ============================================
    CTA BUTTON HOVER EFFECT
    ============================================
    */

    const interactiveCards =
        document.querySelectorAll(
            ".feature-card, .process-card, .about-card, .care-card"
        );


    interactiveCards.forEach(
        function (card) {

            card.addEventListener(
                "mouseenter",
                function () {

                    card.classList.add(
                        "card-hover"
                    );

                }
            );


            card.addEventListener(
                "mouseleave",
                function () {

                    card.classList.remove(
                        "card-hover"
                    );

                }
            );

        }
    );



    /*
    ============================================
    NAVBAR SCROLL EFFECT
    ============================================
    */

    const navbar =
        document.querySelector(
            ".navbar"
        );


    if (navbar) {

        window.addEventListener(
            "scroll",
            function () {

                if (window.scrollY > 20) {

                    navbar.classList.add(
                        "navbar-scrolled"
                    );

                }

                else {

                    navbar.classList.remove(
                        "navbar-scrolled"
                    );

                }

            }
        );

    }



    /*
    ============================================
    BUTTON RIPPLE EFFECT
    ============================================
    */

    const buttons =
        document.querySelectorAll(
            ".primary-button, .secondary-button, .nav-button, .analyze-button, .logout-button"
        );


    buttons.forEach(
        function (button) {

            button.addEventListener(
                "click",
                function (event) {


                    const ripple =
                        document.createElement(
                            "span"
                        );


                    ripple.classList.add(
                        "button-ripple"
                    );


                    const rect =
                        button.getBoundingClientRect();


                    const size =
                        Math.max(
                            rect.width,
                            rect.height
                        );


                    ripple.style.width =
                        size + "px";


                    ripple.style.height =
                        size + "px";


                    ripple.style.left =
                        (
                            event.clientX -
                            rect.left -
                            size / 2
                        ) + "px";


                    ripple.style.top =
                        (
                            event.clientY -
                            rect.top -
                            size / 2
                        ) + "px";


                    button.appendChild(
                        ripple
                    );


                    setTimeout(
                        function () {

                            ripple.remove();

                        },
                        600
                    );

                }
            );

        }
    );



    /*
    ============================================
    IMAGE PREVIEW RESET
    ============================================
    */

    if (imagePreview) {

        imagePreview.addEventListener(
            "error",
            function () {

                imagePreview.src = "";

                if (imagePreviewContainer) {

                    imagePreviewContainer.classList.remove(
                        "show"
                    );

                }

            }
        );

    }



    /*
    ============================================
    CONSOLE MESSAGE
    ============================================
    */

    console.log(
        "EyeCare AI interface initialized successfully."
    );

});