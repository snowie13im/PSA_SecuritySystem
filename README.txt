=============================================================================
  face_recognition.py — Main entry point
  ──────────────────────────────────────────────────────────────────────────
  Responsibilities of this file:
    • Management menu  (list / delete people, start camera, quit)
    • Webcam loop      (detect faces, run LBPH prediction, draw overlays)
    • Unknown-face alerting  (fires events → registration.py handles them)
    • Model loading    (reads the model trained by registration.py)

  Run with:
      python face_recognition.py

  Controls while the camera is active:
      Q  →  quit
      M  →  return to management menu
      R  →  register the unknown face shown in the pop-up
      I  →  ignore the unknown face

  Dependencies:
      pip install opencv-contrib-python

 =============================================================================
  registration.py — Registration subsystem
  
  Everything related to adding a new person to the database lives here:

    • Thread-safe frame sharing  (set_frame / get_frame)
    • Shared concurrency primitives  (event_queue, registration_busy)
    • In-window name input  (_ask_name_in_window)
    • Guided 6-pose capture  (register_guided)
    • Background registration thread  (registration_thread)

  face_recognition.py imports the public symbols:
      set_frame, event_queue, registration_busy, registration_thread

  This module has NO knowledge of the main recognition loop — all
  communication happens through the shared queue and Event flag.
 ========================================================================
 
 =============================================================================
  config.py — Shared configuration
  ──────────────────────────────────────────────────────────────────────────
  Single source of truth for every constant used across the project.
  Both face_recognition.py and registration.py import from here, so
  changing a value once is reflected everywhere instantly.
 ============================================================================