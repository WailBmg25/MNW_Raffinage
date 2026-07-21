"use client";

import { useEffect, useRef } from "react";
import { animate, motion, useMotionValue, useTransform } from "framer-motion";

interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  className?: string;
}

export function AnimatedNumber({
  value,
  decimals = 0,
  suffix = "",
  prefix = "",
  className,
}: AnimatedNumberProps) {
  const motionValue = useMotionValue(value);
  const rounded = useTransform(motionValue, (latest) =>
    `${prefix}${latest.toLocaleString("fr-FR", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })}${suffix}`
  );
  const previous = useRef(value);

  useEffect(() => {
    const controls = animate(previous.current, value, {
      duration: 0.6,
      ease: "easeOut",
      onUpdate: (latest) => motionValue.set(latest),
    });
    previous.current = value;
    return () => controls.stop();
  }, [value, motionValue]);

  return (
    <motion.span className={`tabular-nums ${className ?? ""}`}>{rounded}</motion.span>
  );
}
