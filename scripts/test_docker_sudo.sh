#!/bin/bash
# Test script to diagnose sudo issues in Docker container

echo "=== Docker Sudo Test ==="
echo ""

echo "1. Current user info:"
echo "   whoami: $(whoami)"
echo "   id: $(id)"
echo ""

echo "2. Check if running in container:"
if [ -f /.dockerenv ]; then
    echo "   Running in Docker container"
else
    echo "   NOT in Docker container"
fi
echo ""

echo "3. Check sudo availability:"
which sudo && echo "   sudo found at: $(which sudo)" || echo "   sudo NOT found"
echo ""

echo "4. Check sudoers config:"
if [ -f /etc/sudoers ]; then
    echo "   /etc/sudoers exists"
    grep -i "hostuser\|NOPASSWD" /etc/sudoers 2>/dev/null || echo "   No hostuser/NOPASSWD entries found"
else
    echo "   /etc/sudoers NOT found"
fi
echo ""

echo "5. Test sudo (should work without password):"
sudo echo "   sudo works!" 2>&1 || echo "   sudo FAILED"
echo ""

echo "6. Test apt-get with sudo:"
sudo apt-get update -qq 2>&1 | head -5 || echo "   apt-get update FAILED"
echo ""

echo "7. Check /var/lib/apt permissions:"
ls -la /var/lib/apt/ 2>&1 | head -5
echo ""

echo "=== Test Complete ==="
