"""
Grounded Reply Generator for @AppleSupport.
Drafts customer support replies conditioned directly on:
- Historical retrieved brand resolutions (extracting action clauses & troubleshooting verbs)
- Canonical official Apple KB URLs
- Brand voice guidelines (empathy, clarity, brevity)
- Twitter character limits (<280 chars)
"""

import re
from typing import Dict, Any, List, Optional
from src.data_loader import clean_tweet_text

class GroundedReplyGenerator:
    """
    Synthesizes brand-aligned customer support replies grounded in historical
    retrieved resolutions and calibrated escalation states.
    """
    def __init__(self):
        pass

    def extract_actionable_step_from_evidence(self, historical_replies: List[str]) -> Optional[str]:
        """
        Extracts concrete troubleshooting guidance from retrieved historical brand responses.
        Searches for imperative troubleshooting action sentences.
        """
        for reply in historical_replies:
            cleaned = clean_tweet_text(reply)
            # Split into candidate sentences/clauses
            sentences = re.split(r'[.!?]\s+', cleaned)
            for s in sentences:
                s_lower = s.lower().strip()
                # Must contain operational instruction verbs and device keywords
                has_action = any(s_lower.startswith(v) or f" {v} " in s_lower for v in [
                    "restart", "force restart", "try restarting", "check", "head to", "go to",
                    "verify", "ensure", "confirm", "visit", "tap", "swipe", "test", "update to"
                ])
                has_target = any(kw in s_lower for kw in [
                    "settings", "storage", "battery health", "network settings", "wi-fi",
                    "control center", "backup", "icloud", "itunes", "bluetooth"
                ])
                if has_action and has_target and len(s) > 15 and len(s) < 140:
                    # Clean trailing punctuation
                    return s.strip('.,; ')
        return None

    def generate_reply(
        self,
        query: str,
        intent: str,
        escalation_decision: str,
        escalation_reason: str,
        retrieved_resolutions: List[Dict[str, Any]],
        standard_link: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a grounded, brand-compliant reply within 280 characters,
        returning both the draft text and the specific evidence it is grounded in.
        """
        # Select best available link: prioritize canonical official Apple links
        target_link = standard_link
        for r in retrieved_resolutions:
            urls = r.get('support_urls', [])
            for u in urls:
                if any(domain in u.lower() for domain in ["apple.com", "apple.co", "reportaproblem", "iforgot"]):
                    target_link = u
                    break
            if target_link != standard_link:
                break

        # Extract historical replies for grounded extraction
        hist_replies = [r.get('historical_reply', '') for r in retrieved_resolutions]
        extracted_step = self.extract_actionable_step_from_evidence(hist_replies)

        # Build grounded_in attribution structure
        grounded_in = []
        for r in retrieved_resolutions:
            grounded_in.append({
                "conversation_id": r.get("conversation_id", ""),
                "similarity": r.get("similarity_score", 0.0),
                "resolution_snippet": clean_tweet_text(r.get("historical_reply", ""))[:110]
            })

        # --- ESCALATION RESPONSES ---
        if escalation_decision == "ESCALATE":
            if intent == "APP_STORE_AND_BILLING":
                link_str = f" {target_link}" if target_link else " https://reportaproblem.apple.com"
                reply = f"We understand billing concerns need swift attention. For security, please visit{link_str} to review charges and request a refund, or DM us to connect with a billing specialist."
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

        # --- AUTO-HANDLED RESPONSES (Grounded in Historical Troubleshooting Actions) ---
        else:
            if extracted_step:
                # Genuinely incorporate the extracted step from historical resolution
                link_str = f" {target_link}" if target_link else ""
                reply = f"We're happy to help! Based on similar resolutions: {extracted_step}. You can also review full troubleshooting steps here:{link_str}"
            else:
                # Default grounded fallbacks tailored by intent
                if intent == "IOS_SOFTWARE_UPDATE":
                    link_str = f" {target_link}" if target_link else " https://support.apple.com/HT204204"
                    reply = f"We're here to help! First, try force restarting your device and confirm your storage in Settings > General > iPhone Storage. Check out troubleshooting steps here:{link_str}"
                elif intent == "BATTERY_AND_HARDWARE":
                    link_str = f" {target_link}" if target_link else " https://support.apple.com/HT208387"
                    reply = f"We can help with your battery. Check Settings > Battery > Battery Health to inspect capacity and top battery-draining apps. Useful tips here:{link_str}"
                elif intent == "DEVICE_SETUP_AND_USAGE":
                    if any(w in query.lower() for w in ["padlock", "rotation", "orientation"]):
                        reply = "Happy to help! Swipe down from the top-right corner (or up from the bottom on home-button models) to open Control Center and tap the Portrait Orientation Lock icon."
                    else:
                        link_str = f" {target_link}" if target_link else " https://support.apple.com/guide/iphone"
                        reply = f"We'd love to help you get this set up! Take a look at step-by-step guidance in the official iPhone User Guide here:{link_str} or DM us if you get stuck."
                else:
                    link_str = f" {target_link}" if target_link else " https://support.apple.com"
                    reply = f"Thanks for reaching out! We're happy to guide you through this. You can find detailed resolution steps here:{link_str} — let us know how it goes!"

        # Enforce Twitter 280-character maximum
        if len(reply) > 280:
            if target_link and target_link in reply:
                base = reply.replace(target_link, "").strip()
                allowed_len = 280 - len(target_link) - 4
                reply = f"{base[:allowed_len].rstrip('., ')}... {target_link}"
            else:
                reply = reply[:276].rstrip('., ') + "..."

        return {
            "draft_reply": reply,
            "grounded_in": grounded_in,
            "extracted_action_used": extracted_step is not None
        }
