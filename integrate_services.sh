#!/bin/bash
# integrate_services.sh - Script to integrate service startup into the main system startup

echo "=== Integrating Services into System Startup ==="

# Check if scripts/start_system.py exists
if [ ! -f "scripts/start_system.py" ]; then
    echo "❌ Error: scripts/start_system.py not found"
    exit 1
fi

# Create a backup of the original file
cp scripts/start_system.py scripts/start_system.py.bak
echo "✅ Created backup of start_system.py"

# Add service startup to start_system.py
cat << 'EOF' > service_startup_code.txt

    # Start essential services (Redis, Prometheus, Redis exporter)
    def start_essential_services(self):
        """Start essential services like Redis, Prometheus, and Redis exporter"""
        self.logger.info("Starting essential services...")
        
        try:
            # Check if start_services.sh exists
            if os.path.exists('/app/start_services.sh'):
                self.logger.info("Running start_services.sh...")
                subprocess.run(['bash', '/app/start_services.sh'], check=True)
                self.logger.info("Essential services started successfully")
                return True
            else:
                self.logger.warning("start_services.sh not found, skipping service startup")
                return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start essential services: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error starting essential services: {e}")
            return False

EOF

# Find the right place to insert the new method
LINE_NUM=$(grep -n "def start_system" scripts/start_system.py | head -1 | cut -d':' -f1)
if [ -z "$LINE_NUM" ]; then
    echo "❌ Error: Could not find 'def start_system' in start_system.py"
    exit 1
fi

# Insert the new method before start_system
sed -i "${LINE_NUM}r service_startup_code.txt" scripts/start_system.py
echo "✅ Added start_essential_services method to start_system.py"

# Now find the right place to call the new method in start_system
START_SYSTEM_LINE=$(grep -n "def start_system" scripts/start_system.py | head -1 | cut -d':' -f1)
INIT_COMPONENTS_LINE=$(grep -n "self.initialize_components" scripts/start_system.py | head -1 | cut -d':' -f1)

if [ -z "$INIT_COMPONENTS_LINE" ]; then
    echo "❌ Error: Could not find 'self.initialize_components' in start_system.py"
    exit 1
fi

# Add the call to start_essential_services before initialize_components
sed -i "${INIT_COMPONENTS_LINE}i\        # Start essential services\n        self.start_essential_services()" scripts/start_system.py
echo "✅ Added call to start_essential_services in start_system method"

# Add import for subprocess if not already present
if ! grep -q "import subprocess" scripts/start_system.py; then
    IMPORT_LINE=$(grep -n "import" scripts/start_system.py | head -1 | cut -d':' -f1)
    sed -i "${IMPORT_LINE}i\import subprocess" scripts/start_system.py
    echo "✅ Added import for subprocess"
fi

# Clean up temporary file
rm service_startup_code.txt

echo "=== Integration Complete ==="
echo "The system will now automatically start Redis, Prometheus, and Redis exporter on startup."
echo "To test the changes, run: ./start.sh"
echo "To revert changes if needed, run: cp scripts/start_system.py.bak scripts/start_system.py"