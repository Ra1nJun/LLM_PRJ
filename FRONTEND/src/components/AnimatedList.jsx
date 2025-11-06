import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import "./AnimatedList.css";

const AnimatedItem = ({ children, delay = 0 }) => {
  const ref = useRef(null);
  const inView = useInView(ref, { amount: 0.3, triggerOnce: false });

  return (
    <motion.div
      ref={ref}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={inView ? { scale: 1, opacity: 1 } : { scale: 0.9, opacity: 0 }}
      transition={{ duration: 0.2, delay }}
      style={{ marginBottom: "0.6rem" }}
    >
      {children}
    </motion.div>
  );
};

const AnimatedList = ({ messages }) => {
  return (
    <div className="scroll-list-container">
      <div className="scroll-list">
        {messages.map((msg, i) => (
          <AnimatedItem key={msg.id || i} delay={0.05 * i}>
            <div
              className={`chat-bubble ${msg.sender}`}
              style={{
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
              }}
            >
              <p>{msg.text}</p>
            </div>
          </AnimatedItem>
        ))}
      </div>
    </div>
  );
};

export default AnimatedList;
