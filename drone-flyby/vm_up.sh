#!/usr/bin/env bash
# Start the serving VM and show what deploy_vm.sh needs: its IP and whether ports 22 and 9053 are open.
#
#   bash vm_up.sh           # start, show IP and ports, open 9053 if it is closed
#   bash vm_up.sh down      # deallocate: no compute cost until the next start
set -uo pipefail
RG=nordic-ai-cup
VM=amigos-ml
PORT=9053

if [ "${1:-}" = down ]; then
    az vm deallocate -g "$RG" -n "$VM" && echo "$VM deallocated"
    exit
fi

echo "== starting $VM"
az vm start -g "$RG" -n "$VM" -o none && echo "started"

echo "== state"
az vm show -d -g "$RG" -n "$VM" --query "{state:powerState, ip:publicIps, size:hardwareProfile.vmSize, location:location}" -o table

# The NSG can sit on the NIC or on its subnet.
NIC=$(az vm show -g "$RG" -n "$VM" --query "networkProfile.networkInterfaces[0].id" -o tsv)
NSG=$(az network nic show --ids "$NIC" --query "networkSecurityGroup.id" -o tsv)
if [ -z "$NSG" ]; then
    SUBNET=$(az network nic show --ids "$NIC" --query "ipConfigurations[0].subnet.id" -o tsv)
    NSG=$(az network vnet subnet show --ids "$SUBNET" --query "networkSecurityGroup.id" -o tsv)
fi
NSG_NAME=$(basename "$NSG")
echo "== inbound rules of $NSG_NAME"
az network nsg rule list -g "$RG" --nsg-name "$NSG_NAME" \
    --query "[?direction=='Inbound'].{name:name, priority:priority, port:destinationPortRange, ports:join(',', destinationPortRanges), source:sourceAddressPrefix, access:access}" -o table

OPEN=$(az network nsg rule list -g "$RG" --nsg-name "$NSG_NAME" \
    --query "[?direction=='Inbound' && access=='Allow' && (destinationPortRange=='$PORT' || contains(destinationPortRanges, '$PORT') || destinationPortRange=='*')].name" -o tsv)
if [ -z "$OPEN" ]; then
    echo "== opening port $PORT"
    az network nsg rule create -g "$RG" --nsg-name "$NSG_NAME" -n "allow-drone-$PORT" --priority 1010 \
        --direction Inbound --access Allow --protocol Tcp --destination-port-ranges "$PORT" -o none && echo "opened $PORT"
else
    echo "== port $PORT already open ($OPEN)"
fi
