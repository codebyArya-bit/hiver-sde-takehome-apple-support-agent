"""
Grounded Reply Generator for @AppleSupport.
Drafts replies conditioned on historical brand resolutions, official Apple KB links,
brand voice guidelines (empathy, clarity, brevity), and Twitter length constraints (<280 chars).
"""

import re
from typing import Dict, Any, List, Optional

class GroundedReplyGenerator:
    """
    Synthesizes brand-aligned customer support replies grounded in historical
    retrieved resolutions and calibrated escalation states.
    """
    def __init__(self):
        pass

    def generate_reply(
        self,
        query: str,
        intent: str,
        escalation_decision: str,
        escalation_reason: str,
        retrieved_resolutions: List[Dict[str, Any]],
        standard_link: Optional[str] = None
    ) -> str:
        """
        Generates a grounded, brand-compliant reply within 280 characters.
        """
        # Select best available link: prefer verified official Apple links or standard official link over stale t.co redirects
        target_link = standard_link
        for r in retrieved_resolutions:
            urls = r.get('support_urls', [])
            for u in urls:
                if any(domain in u.lower() for domain in ["apple.com", "apple.co", "reportaproblem", "iforgot"]):
                    target_link = u
                    break
            if target_link != standard_link:
                break

        # --- ESCALATION RESPONSES ---
        if escalation_decision == "ESCALATE":
            if intent == "APP_STORE_AND_BILLING":
                link_str = f" {target_link}" if target_link else " https://reportaproblem.apple.com"
                reply = f"We understand billing concerns need swift attention. For security, please visit{link_str} to review charges and request a refund, or DM us to connect with a specialist."
            elif intent == "APPLE_ID_AND_ICLOUD":
                link_str = f" {target_link}" if target_link else " https://iforgot.apple.com"
                reply = f"Account security is our top priority. Please head to{link_str} to securely verify and recover your Apple ID. If you're still locked out, DM us to assist further."
            elif intent == "BATTERY_AND_HARDWARE":
                link_str = f" {target_link}" if target_link else " https://support.apple.com/repair"
                reply = f"We want to ensure your device is running safely. For physical hardware inspection or a Genius Bar appointment, check:{link_str} or DM us your zip code."
            elif intent == "CUSTOMER_FEEDBACK_COMPLAINT":
                reply = "We're truly sorry for the frustration this has caused. We want to make this right—please DM us with your device model, case number (if any), and details so a senior specialist can assist."
            else:
                reply = "We'd love to look into this with you directly. Please send us a DM with your device model, current iOS version, and any details so we can assist: https://twitter.com/messages/compose"

        # --- AUTO-HANDLED RESPONSES (Grounded Self-Service) ---
        else:
            if intent == "IOS_SOFTWARE_UPDATE":
                link_str = f" {target_link}" if target_link else " https://support.apple.com/HT204204"
                reply = f"We're here to help! First, try force restarting your device and confirm your storage in Settings > General > iPhone Storage. Check out troubleshooting steps here:{link_str}"
            elif intent == "BATTERY_AND_HARDWARE":
                link_str = f" {target_link}" if target_link else " https://support.apple.com/HT208387"
                reply = f"We can help with your battery. Check Settings > Battery > Battery Health to inspect capacity and top battery-draining apps. Useful tips here:{link_str}"
            elif intent == "DEVICE_SETUP_AND_USAGE":
                # Check for orientation lock
                if "padlock" in query.lower() or "lock" in query.lower() or "rotation" in query.lower():
                    reply = "Happy to help! Swipe down from the top-right corner (or up from the bottom on home-button models) to open Control Center and tap the Portrait Orientation Lock icon."
                else:
                    link_str = f" {target_link}" if target_link else " https://support.apple.com/guide/iphone"
                    reply = f"We'd love to help you get this set up! Take a look at step-by-step guidance in the official iPhone User Guide here:{link_str} or DM us if you get stuck."
            else:
                link_str = f" {target_link}" if target_link else " https://support.apple.com"
                reply = f"Thanks for reaching out! We're happy to guide you through this. You can find detailed resolution steps here:{link_str} — let us know how it goes!"

        # Enforce Twitter 280-character maximum
        if len(reply) > 280:
            # Smart truncate preserving link at the end if present
            if target_link and target_link in reply:
                base = reply.replace(target_link, "").strip()
                allowed_len = 280 - len(target_link) - 4
                reply = f"{base[:allowed_len].rstrip('., ')}... {target_link}"
            else:
                reply = reply[:276].rstrip('., ') + "..."

        return reply
